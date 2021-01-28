//====- LowerToAffineLoops.cpp - Partial lowering from Toy to Affine+Std --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a combination of
// GPU and standard operations. This lowering expects that all calls
// have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ToyToGPU RewritePatterns
//===----------------------------------------------------------------------===//

/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// to use `print_memref_f32`, we need to do registering with `mcuMemHostRegisterFloat`
static FlatSymbolRefAttr 
getOrInsertMcuMemHostRegisterFloat(PatternRewriter &rewriter,
                                   ModuleOp module,
                                   MemRefType type) {
  auto *context = module.getContext();
  if (module.lookupSymbol<mlir::FuncOp>("mcuMemHostRegisterFloat"))
    return SymbolRefAttr::get("mcuMemHostRegisterFloat", context);
    
  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  
  auto fnType = FunctionType::get({UnrankedMemRefType::get(type.getElementType(), 0)},
                                  ArrayRef<Type>({}),
                                  context);
  rewriter.create<FuncOp>(module.getLoc(), "mcuMemHostRegisterFloat", fnType);
  return SymbolRefAttr::get("mcuMemHostRegisterFloat", context);
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter,
                                   ModuleOp parentModule) {
  auto alloc = rewriter.create<mlir::AllocOp>(loc, type);
  
  auto mcuMemHostRegisterFloatRef = 
    getOrInsertMcuMemHostRegisterFloat(rewriter, parentModule, type);

  auto unrankedMemRefType = UnrankedMemRefType::get(type.getElementType(), 0);
  auto memrefCast = rewriter.create<mlir::MemRefCastOp>(loc, alloc, unrankedMemRefType);
  auto registerCall = rewriter.create<mlir::CallOp>(loc, mcuMemHostRegisterFloatRef,
                                                    ArrayRef<Type>({}),
                                                    ArrayRef<Value>({memrefCast}));

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc.getOperation()->getBlock();
  registerCall.getOperation()->moveBefore(&parentBlock->front());
  memrefCast.getOperation()->moveBefore(&parentBlock->front());
  alloc.getOperation()->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<mlir::DeallocOp>(loc, alloc);
  dealloc.getOperation()->moveBefore(&parentBlock->back());
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, an range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for the iteration. It returns a value to store at the
/// current index of the iteration.
using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter, op->getParentOfType<ModuleOp>());

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
  buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function with the rewriter, the memref operands,
        // and the loop induction variables. This function will return the value
        // to store at the current index.
        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<AffineStoreOp>(loc, valueToStore, alloc, ivs);
      });

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

namespace {
//===----------------------------------------------------------------------===//
// ToyToGPU RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
    : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
          ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Insert an allocation and deallocation for the result of this operation.
    auto tensorType = (*op->result_type_begin()).cast<TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter, op->getParentOfType<ModuleOp>());

    // get the shape to set gridSize and blockSize
    auto shape = tensorType.getShape().vec();
    auto const_1 = rewriter.create<ConstantIndexOp>(loc, 1);
    auto shapeX = rewriter.create<ConstantIndexOp>(loc, shape[0]);
    auto shapeY = rewriter.create<ConstantIndexOp>(loc, shape[1]);

    gpu::KernelDim3 gridSizes = {shapeX, const_1, const_1};
    gpu::KernelDim3 blockSizes = {shapeY, const_1, const_1};
    
    auto launchOp = rewriter.create<gpu::LaunchOp>(loc,
                                                  gridSizes.x, gridSizes.y, gridSizes.z,
                                                  blockSizes.x, blockSizes.y, blockSizes.z);
    
    //=== start and fill the body of gpu launchOp
    rewriter.setInsertionPointToStart(&launchOp.body().front());

    typename BinaryOp::Adaptor binaryAdaptor(operands);
    ValueRange indices({launchOp.getBlockIds().x, launchOp.getThreadIds().x});

    auto loadedLhs = rewriter.create<mlir::LoadOp>(loc, binaryAdaptor.lhs(), indices);
    auto loadedRhs = rewriter.create<mlir::LoadOp>(loc, binaryAdaptor.rhs(), indices);
    auto result =  rewriter.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
    auto store = rewriter.create<mlir::StoreOp>(loc, result, alloc, indices);
    auto terminator =  rewriter.create<gpu::TerminatorOp>(loc);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

using AddOpLowering = BinaryOpLowering<toy::AddOp, mlir::AddFOp>;
using SubOpLowering = BinaryOpLowering<toy::SubOp, mlir::SubFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, mlir::MulFOp>;


//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: MatrixMul operations
//===----------------------------------------------------------------------===//

struct MatrixMulOpLowering : public ConversionPattern {
  MatrixMulOpLowering(MLIRContext *ctx)
    : ConversionPattern(toy::MatrixMulOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
          ConversionPatternRewriter &rewriter) const final {
  
  // Modified from lowerOpToLoops
  auto loc = op->getLoc();
  auto tensorType = (*op->result_type_begin()).cast<TensorType>();
  
  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter, op->getParentOfType<ModuleOp>());

  // get the shape to set gridSize and blockSize
  auto shape = tensorType.getShape().vec();
  auto const_1 = rewriter.create<ConstantIndexOp>(loc, 1);
  auto shapeX = rewriter.create<ConstantIndexOp>(loc, shape[0]);
  auto shapeY = rewriter.create<ConstantIndexOp>(loc, shape[1]);
  gpu::KernelDim3 gridSizes = {shapeX, shapeY, const_1};
  gpu::KernelDim3 blockSizes = {shapeY, const_1, const_1};

  auto launchOp = rewriter.create<gpu::LaunchOp>(loc,
                                              gridSizes.x, gridSizes.y, gridSizes.z,
                                              blockSizes.x, blockSizes.y, blockSizes.z);

  typename toy::MatrixMulOp::Adaptor MatrixMulAdaptor(operands);
  ValueRange indicesLhs({launchOp.getBlockIds().x, launchOp.getThreadIds().x});
  ValueRange indicesRhs({launchOp.getThreadIds().x, launchOp.getBlockIds().y});
  ValueRange indicesResult({launchOp.getBlockIds().x, launchOp.getBlockIds().y});

  // mul and reduce operation in gpu
  rewriter.setInsertionPointToStart(&launchOp.body().front());
  auto LoadedLhs = rewriter.create<mlir::LoadOp>(loc, MatrixMulAdaptor.lhs(), indicesLhs);
  auto LoadedRhs = rewriter.create<mlir::LoadOp>(loc, MatrixMulAdaptor.rhs(), indicesRhs);
  auto mulResult = rewriter.create<mlir::MulFOp>(loc, LoadedLhs, LoadedRhs);
  auto ReducedResult = rewriter.create<gpu::AllReduceOp>(loc, 
                                                         rewriter.getF32Type(), 
                                                         mulResult, 
                                                         StringAttr::get("add",rewriter.getContext()));
  auto StoredResult = rewriter.create<mlir::StoreOp>(loc, ReducedResult, alloc, indicesResult);
  auto terminator =  rewriter.create<gpu::TerminatorOp>(loc);
  rewriter.setInsertionPointToEnd(&launchOp.body().front());

  // // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
  return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<toy::ConstantOp> {
  using OpRewritePattern<toy::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::ConstantOp op,
                PatternRewriter &rewriter) const final {
  DenseElementsAttr constantValue = op.value();
  Location loc = op.getLoc();

  // When lowering the constant operation, we allocate and assign the constant
  // values to a corresponding memref allocation.
  auto tensorType = op.getType().cast<TensorType>();
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter, op.getParentOfType<ModuleOp>());

  // We will be generating constant indices up-to the largest dimension.
  // Create these constants up-front to avoid large amounts of redundant
  // operations.
  auto valueShape = memRefType.getShape();
  SmallVector<Value, 8> constantIndices;

  if (!valueShape.empty()) {
    for (auto i : llvm::seq<int64_t>(
        0, *std::max_element(valueShape.begin(), valueShape.end())))
     constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, i));
  } else {
    // This is the case of a tensor of rank 0.
    constantIndices.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
  }

  // The constant operation represents a multi-dimensional constant, so we
  // will need to generate a store for each of the elements. The following
  // functor recursively walks the dimensions of the constant shape,
  // generating a store when the recursion hits the base case.
  SmallVector<Value, 2> indices;
  auto valueIt = constantValue.getValues<FloatAttr>().begin();
  std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
    // The last dimension is the base case of the recursion, at this point
    // we store the element at the given index.
    if (dimension == valueShape.size()) {
    rewriter.create<mlir::StoreOp>(
      loc, rewriter.create<mlir::ConstantOp>(loc, *valueIt++), alloc,
      llvm::makeArrayRef(indices));
    return;
    }

    // Otherwise, iterate over the current dimension and add the indices to
    // the list.
    for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
    indices.push_back(constantIndices[i]);
    storeElements(dimension + 1);
    indices.pop_back();
    }
  };

  // Start the element storing recursion from the first dimension.
  storeElements(/*dimension=*/0);

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
  return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToGPU RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<toy::ReturnOp> {
  using OpRewritePattern<toy::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(toy::ReturnOp op,
                PatternRewriter &rewriter) const final {
  // During this lowering, we expect that all function calls have been
  // inlined.
  if (op.hasOperand())
    return failure();

  // We lower "toy.return" directly to "std.return".
  rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op);
  return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx)
    : ConversionPattern(toy::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
          ConversionPatternRewriter &rewriter) const final {
  auto loc = op->getLoc();
  lowerOpToLoops(op, operands, rewriter,
           [loc](OpBuilder &builder, ValueRange memRefOperands,
             ValueRange loopIvs) {
           // Generate an adaptor for the remapped operands of the
           // TransposeOp. This allows for using the nice named
           // accessors that are generated by the ODS.
           toy::TransposeOpAdaptor transposeAdaptor(memRefOperands);
           Value input = transposeAdaptor.input();

           // Transpose the elements by generating a load from the
           // reverse indices.
           SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
           return builder.create<mlir::AffineLoadOp>(loc, input,
                             reverseIvs);
           });
  return success();
  }
};


//===----------------------------------------------------------------------===//
// ToyToGPU RewritePatterns: print operations
//===----------------------------------------------------------------------===//

class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(toy::PrintOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {                 
      auto loc = op->getLoc();

      // // Insert an allocation and deallocation for the result of this operation.
      auto unrankedMemRefType = UnrankedMemRefType::get(rewriter.getF32Type(), 0);
      auto memrefCast =  rewriter.create<mlir::MemRefCastOp>(loc, operands[0], unrankedMemRefType);

      // // get or insert Print
      ModuleOp parentModule = op->getParentOfType<ModuleOp>();
      auto printRef = getOrInsertPrint(rewriter, parentModule);
      rewriter.create<mlir::CallOp>(loc, printRef, ArrayRef<Type>({}), ArrayRef<Value>({memrefCast}));
      
      rewriter.eraseOp(op);
      return success();    
  }

private:
  /// Return a symbol reference to the print_memref_f32 function, 
  /// inserting it into the module if necessary.
  static FlatSymbolRefAttr getOrInsertPrint(PatternRewriter &rewriter,
                                            ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<mlir::FuncOp>("print_memref_f32"))
      return SymbolRefAttr::get("print_memref_f32", context);
      
    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    
    auto fnType = FunctionType::get({UnrankedMemRefType::get(rewriter.getF32Type(), 0)},
                                    ArrayRef<Type>({}), context);
    rewriter.create<mlir::FuncOp>(module.getLoc(), "print_memref_f32", fnType);
    return SymbolRefAttr::get("print_memref_f32", context);
  }
};

} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// ToyToGPULoweringPass
//===----------------------------------------------------------------------===//

namespace {
struct ToyToGPULoweringPass
  : public PassWrapper<ToyToGPULoweringPass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void ToyToGPULoweringPass::runOnFunction() {
  auto function = getFunction();

  // We only lower the main function as we expect that all other functions have
  // been inlined.
  if (function.getName() != "main")
  return;

  // Verify that the given main has no inputs and results.
  if (function.getNumArguments() || function.getType().getNumResults()) {
  function.emitError("expected 'main' to have 0 inputs and 0 results");
  return signalPassFailure();
  }

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine` and `Standard` dialects.
  target.addLegalDialect<gpu::GPUDialect, mlir::AffineDialect, mlir::StandardOpsDialect>();
  target.addLegalOp<mlir::FuncOp>();
  target.addIllegalDialect<toy::ToyDialect>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  OwningRewritePatternList patterns;
  patterns.insert<ConstantOpLowering, 
                  AddOpLowering, SubOpLowering, MulOpLowering, MatrixMulOpLowering,
                  TransposeOpLowering,
                  PrintOpLowering,
                  ReturnOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(getFunction(), target, patterns)))
  signalPassFailure();
}

/// Create a pass for lowering operations in the `GPU` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::toy::createLowerToGPUPass() {
  return std::make_unique<ToyToGPULoweringPass>();
}
