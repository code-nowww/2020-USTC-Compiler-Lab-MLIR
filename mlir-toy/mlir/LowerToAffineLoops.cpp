//====- LowerToAffineLoops.cpp - Partial lowering from Toy to Affine+Std --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a combination of
// affine loops and standard operations. This lowering expects that all calls
// have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/IR/Constants.h"
#include <vector>
using namespace std;
using namespace mlir;


//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns
//===----------------------------------------------------------------------===//

/// Convert the given TensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc.getOperation()->getBlock();
  alloc.getOperation()->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<DeallocOp>(loc, alloc);
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
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

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
// ToyToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](OpBuilder &builder, ValueRange memRefOperands,
              ValueRange loopIvs) {
          // Generate an adaptor for the remapped operands of the BinaryOp. This
          // allows for using the nice named accessors that are generated by the
          // ODS.
          typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

          // Generate loads for the element of 'lhs' and 'rhs' at the inner
          // loop.
          auto loadedLhs =
              builder.create<AffineLoadOp>(loc, binaryAdaptor.lhs(), loopIvs);
          auto loadedRhs =
              builder.create<AffineLoadOp>(loc, binaryAdaptor.rhs(), loopIvs);

          // Create the binary operation performed on the loaded values.
          return builder.create<LoweredBinaryOp>(loc, loadedLhs, loadedRhs);
        });
    return success();
  }
};
using AddOpLowering = BinaryOpLowering<toy::AddOp, AddFOp>;
using SubOpLowering = BinaryOpLowering<toy::SubOp, SubFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, MulFOp>;
using CmpOpLowering = BinaryOpLowering<toy::CmpOp, AddFOp>;
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
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    SmallVector<int64_t, 4> LowerBounds(2, /*Value=*/0);
    SmallVector<int64_t, 4> Steps(2, /*Value=*/1);
    
    buildAffineLoopNest(
        rewriter, loc, LowerBounds, tensorType.getShape(), Steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          const APFloat zero(0.0);
          nestedBuilder.create<AffineStoreOp>(loc, rewriter.create<ConstantFloatOp>(loc, zero, nestedBuilder.getF64Type()), alloc, ivs);
        });
    
    SmallVector<int64_t, 4> loopShape;
    SmallVector<int64_t, 4> loopLowerBounds(3, /*Value=*/0);
    SmallVector<int64_t, 4> steps(3, /*Value=*/1); 

    loopShape.push_back(op->getOperand(0).getType().cast<TensorType>().getShape().vec()[0]);
    loopShape.push_back(op->getOperand(0).getType().cast<TensorType>().getShape().vec()[1]);
    loopShape.push_back(op->getOperand(1).getType().cast<TensorType>().getShape().vec()[1]);
    
    buildAffineLoopNest(
        rewriter, loc, loopLowerBounds, loopShape, steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {

          toy::MatrixMulOpAdaptor MatrixMulAdaptor(operands);
          Value lhs = MatrixMulAdaptor.lhs();
          Value rhs = MatrixMulAdaptor.rhs();

          SmallVector<AffineExpr, 2> lhsExprs, rhsExprs, resultExprs;

          lhsExprs.push_back(getAffineDimExpr(0, nestedBuilder.getContext()));
          lhsExprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()));
          rhsExprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()));
          rhsExprs.push_back(getAffineDimExpr(2, nestedBuilder.getContext()));
          resultExprs.push_back(getAffineDimExpr(0, nestedBuilder.getContext()));
          resultExprs.push_back(getAffineDimExpr(2, nestedBuilder.getContext()));

          auto LoadedLhs = nestedBuilder.create<AffineLoadOp>(loc, lhs, AffineMap::get(3, 0, lhsExprs, nestedBuilder.getContext()), ivs);
          auto LoadedRhs = nestedBuilder.create<AffineLoadOp>(loc, rhs, AffineMap::get(3, 0, rhsExprs, nestedBuilder.getContext()), ivs);
          auto mulResult = nestedBuilder.create<MulFOp>(loc, LoadedLhs, LoadedRhs);
          auto LoadedResult = nestedBuilder.create<AffineLoadOp>(loc, alloc, AffineMap::get(3, 0, resultExprs, nestedBuilder.getContext()), ivs);
          auto addResult = nestedBuilder.create<AddFOp>(loc, mulResult, LoadedResult);
          nestedBuilder.create<AffineStoreOp>(loc, addResult, alloc, AffineMap::get(3, 0, resultExprs, nestedBuilder.getContext()), ivs);
        });

    // Replace this operation with the generated alloc.
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
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

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
        rewriter.create<AffineStoreOp>(
            loc, rewriter.create<ConstantOp>(loc, *valueIt++), alloc,
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
// ToyToAffine RewritePatterns: Return operations
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
    rewriter.replaceOpWithNewOp<ReturnOp>(op);
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
                     return builder.create<AffineLoadOp>(loc, input,
                                                         reverseIvs);
                   });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: valid mode Convolution operations
//===----------------------------------------------------------------------===//

struct ConvValidOpLowering : public ConversionPattern {
  ConvValidOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::ConvValidOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
  //getOperand(unsigned idx)  
    auto loc = op->getLoc();
    auto targetType = op->getOperand(0).getType().cast<TensorType>();
    auto kernelType = op->getOperand(1).getType().cast<TensorType>();
    auto resultType = (*op->result_type_begin()).cast<TensorType>();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto memRefType = convertTensorToMemRef(resultType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // Create a nest of affine loops, with one loop per dimension of the shape.
    // The buildAffineLoopNest function takes a callback that is used to construct
    // the body of the innermost loop given a builder, a location and a range of
    // loop induction variables.
    SmallVector<int64_t, 4> lowerBounds(4, /*Value=*/0);
    SmallVector<int64_t, 4> upperBounds;
    SmallVector<int64_t, 4> steps(4, /*Value=*/1);
    upperBounds.push_back(resultType.getShape().front());
    upperBounds.push_back(resultType.getShape().back());
    upperBounds.push_back(kernelType.getShape().front());
    upperBounds.push_back(kernelType.getShape().back());

    buildAffineLoopNest(
      rewriter, loc, lowerBounds, upperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {

        toy::ConvValidOpAdaptor ConvValidAdaptor(operands);
        Value target = ConvValidAdaptor.target();
        Value kernel = ConvValidAdaptor.kernel();
        
        SmallVector<AffineExpr, 2> inputExprs, kernelExprs, resultExprs;
        /// getAffineDimExpr:position:Position of this identifier in the argument list.
        inputExprs.push_back(getAffineDimExpr(0, nestedBuilder.getContext()) + getAffineDimExpr(2, nestedBuilder.getContext()));
        inputExprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()) + getAffineDimExpr(3, nestedBuilder.getContext()));
        kernelExprs.push_back(getAffineConstantExpr(kernelType.getShape().vec()[0] - 1, nestedBuilder.getContext()) -
                              getAffineDimExpr(2, nestedBuilder.getContext()));
        kernelExprs.push_back(getAffineConstantExpr(kernelType.getShape().vec()[1] - 1, nestedBuilder.getContext()) -
                              getAffineDimExpr(3, nestedBuilder.getContext()));
        resultExprs.push_back(getAffineDimExpr(0, nestedBuilder.getContext()));
        resultExprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()));
        auto LoadedInput = nestedBuilder.create<AffineLoadOp>(loc, target, AffineMap::get(4, 0, inputExprs, nestedBuilder.getContext()), ivs);
        auto LoadedKernel = nestedBuilder.create<AffineLoadOp>(loc, kernel, AffineMap::get(4, 0, kernelExprs, nestedBuilder.getContext()), ivs);
        auto mulResult = nestedBuilder.create<MulFOp>(loc, LoadedInput, LoadedKernel);
        auto LoadedResult = nestedBuilder.create<AffineLoadOp>(loc, alloc, AffineMap::get(4, 0, resultExprs, nestedBuilder.getContext()), ivs);
        auto addResult = nestedBuilder.create<AddFOp>(loc, mulResult, LoadedResult);
        nestedBuilder.create<AffineStoreOp>(loc, addResult, alloc, AffineMap::get(4, 0, resultExprs, nestedBuilder.getContext()), ivs);
      });

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};


//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: valid mode Convolution operations
//===----------------------------------------------------------------------===//

struct FillFullOpLowering : public ConversionPattern {
  FillFullOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::FillFullOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
  //getOperand(unsigned idx)  
    auto loc = op->getLoc();
    auto targetType = op->getOperand(0).getType().cast<TensorType>();
    auto kernelType = op->getOperand(1).getType().cast<TensorType>();
    auto resultType = (*op->result_type_begin()).cast<TensorType>();

    auto memRefType = convertTensorToMemRef(resultType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    SmallVector<int64_t, 2> firstlowerBounds(2, /*Value=*/0);
    SmallVector<int64_t, 2> steps(2, /*Value=*/1);

    buildAffineLoopNest(
      rewriter, loc, firstlowerBounds, resultType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {

        //toy::ConvValidOpAdaptor ConvValidAdaptor(operands);
        //Value target = ConvValidAdaptor.target();
        const APFloat zero(0.0);
        nestedBuilder.create<AffineStoreOp>(loc,rewriter.create<ConstantFloatOp>(loc,zero,nestedBuilder.getF64Type()),alloc, ivs); 
      });

    SmallVector<int64_t, 2> secondlowerBounds;
    SmallVector<int64_t, 2> secondupperBounds;
    secondlowerBounds.push_back(kernelType.getShape().vec()[0]-1);
    secondlowerBounds.push_back(kernelType.getShape().vec()[1]-1);
    secondupperBounds.push_back(kernelType.getShape().vec()[0] + targetType.getShape().vec()[0] - 1);
    secondupperBounds.push_back(kernelType.getShape().vec()[1] + targetType.getShape().vec()[1] - 1);

    buildAffineLoopNest(
      rewriter, loc, secondlowerBounds, secondupperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {

        toy::FillFullOpAdaptor FillFullAdaptor(operands);
        Value target = FillFullAdaptor.target();
        
        SmallVector<AffineExpr, 2> exprs;

        exprs.push_back(getAffineDimExpr(0, nestedBuilder.getContext()) -
                        getAffineConstantExpr(kernelType.getShape().vec()[0] - 1, nestedBuilder.getContext()));
        exprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()) -
                        getAffineConstantExpr(kernelType.getShape().vec()[1] - 1, nestedBuilder.getContext()));
        
        auto LoadedTarget = nestedBuilder.create<AffineLoadOp>(loc, target, AffineMap::get(2, 0, exprs, nestedBuilder.getContext()), ivs);
        nestedBuilder.create<AffineStoreOp>(loc, LoadedTarget, alloc, ivs);
      });
    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: valid mode Convolution operations
//===----------------------------------------------------------------------===//

struct FillSomeOpLowering : public ConversionPattern {
  FillSomeOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::FillSomeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
  //getOperand(unsigned idx)  
    auto loc = op->getLoc();
    auto targetType = op->getOperand(0).getType().cast<TensorType>();
    auto kernelType = op->getOperand(1).getType().cast<TensorType>();
    auto resultType = (*op->result_type_begin()).cast<TensorType>();

    auto memRefType = convertTensorToMemRef(resultType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    SmallVector<int64_t, 2> firstlowerBounds(2, /*Value=*/0);
    SmallVector<int64_t, 2> steps(2, /*Value=*/1);

    buildAffineLoopNest(
      rewriter, loc, firstlowerBounds, resultType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {

        //toy::ConvValidOpAdaptor ConvValidAdaptor(operands);
        //Value target = ConvValidAdaptor.target();
        const APFloat zero(0.0);
        nestedBuilder.create<AffineStoreOp>(loc,rewriter.create<ConstantFloatOp>(loc,zero,nestedBuilder.getF64Type()),alloc, ivs); 
      });

    SmallVector<int64_t, 2> secondlowerBounds;
    SmallVector<int64_t, 2> secondupperBounds;
    secondlowerBounds.push_back((kernelType.getShape().vec()[0] - 1) / 2);
    secondlowerBounds.push_back((kernelType.getShape().vec()[1] - 1) / 2);
    secondupperBounds.push_back((kernelType.getShape().vec()[0] - 1) / 2 + targetType.getShape().vec()[0] );
    secondupperBounds.push_back((kernelType.getShape().vec()[1] - 1) / 2 + targetType.getShape().vec()[1] );

    buildAffineLoopNest(
      rewriter, loc, secondlowerBounds, secondupperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {

        toy::FillFullOpAdaptor FillFullAdaptor(operands);
        Value target = FillFullAdaptor.target();
        
        SmallVector<AffineExpr, 2> exprs;

        exprs.push_back(getAffineDimExpr(0, nestedBuilder.getContext()) -
                        getAffineConstantExpr((kernelType.getShape().vec()[0] - 1) / 2, nestedBuilder.getContext()));
        exprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()) -
                        getAffineConstantExpr((kernelType.getShape().vec()[1] - 1) / 2, nestedBuilder.getContext()));
        
        auto LoadedTarget = nestedBuilder.create<AffineLoadOp>(loc, target, AffineMap::get(2, 0, exprs, nestedBuilder.getContext()), ivs);
        nestedBuilder.create<AffineStoreOp>(loc, LoadedTarget, alloc, ivs);
      });
    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: LU decomposition operations
//===----------------------------------------------------------------------===//

struct LUOpLowering : public ConversionPattern {
  LUOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::LUOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
   
    auto loc = op->getLoc();
    auto targetType = op->getOperand(0).getType().cast<TensorType>();
    auto resultType = (*op->result_type_begin()).cast<TensorType>();
    int64_t looplen = targetType.getShape().vec()[0];
    auto memRefType = convertTensorToMemRef(resultType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    SmallVector<int64_t, 2> firstlowerBounds(2, /*Value=*/0);
    SmallVector<int64_t, 2> steps(2, /*Value=*/1);

    buildAffineLoopNest(
      rewriter, loc, firstlowerBounds, resultType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {

        const APFloat zero(0.0);
        const APFloat One(1.0);
        nestedBuilder.create<AffineStoreOp>(loc,rewriter.create<ConstantFloatOp>(loc,zero,nestedBuilder.getF64Type()),alloc, ivs); 
      });

    SmallVector<int64_t, 2> secondlowerBounds;
    SmallVector<int64_t, 2> secondupperBounds;
    secondlowerBounds.push_back(0);
    secondlowerBounds.push_back(0);
    secondupperBounds.push_back(looplen);
    secondupperBounds.push_back(1);

    int64_t manualIter = 0;
    buildAffineLoopNest(
      rewriter, loc, secondlowerBounds, secondupperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {

        const APFloat One(1.0);
        SmallVector<AffineExpr, 2> exprs;
        exprs.push_back(getAffineDimExpr(0, nestedBuilder.getContext()));
        exprs.push_back(getAffineDimExpr(0, nestedBuilder.getContext()));

        auto oneexpr = rewriter.create<ConstantFloatOp>(loc,One,nestedBuilder.getF64Type());
        nestedBuilder.create<AffineStoreOp>(loc, oneexpr, alloc, AffineMap::get(2, 0, exprs, nestedBuilder.getContext()), ivs);
        
      });
    
    SmallVector<int64_t, 2> thirdlowerBounds(2, 0);
    SmallVector<int64_t, 2> thirdupperBounds;
    SmallVector<int64_t, 2> thirdsteps(2, 1);
    thirdupperBounds.push_back(1);
    thirdupperBounds.push_back(looplen);
    buildAffineLoopNest(
      rewriter, loc, thirdlowerBounds, thirdupperBounds, thirdsteps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        SmallVector<AffineExpr, 2> zeroexprs, Uexprs, Lexprs;
        toy::LUOpAdaptor LUAdaptor(operands);
        Value target = LUAdaptor.input();

        zeroexprs.push_back(getAffineConstantExpr(0, nestedBuilder.getContext()));
        zeroexprs.push_back(getAffineConstantExpr(0, nestedBuilder.getContext()));
        Uexprs.push_back(getAffineConstantExpr(looplen, nestedBuilder.getContext()));
        Uexprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()));
        Lexprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()));
        Lexprs.push_back(getAffineConstantExpr(0, nestedBuilder.getContext()));

        auto uzero = nestedBuilder.create<AffineLoadOp>(loc, target, AffineMap::get(2, 0, zeroexprs, nestedBuilder.getContext()), ivs);
        auto LoadedTarget = nestedBuilder.create<AffineLoadOp>(loc, target, ivs);
        nestedBuilder.create<AffineStoreOp>(loc, LoadedTarget, alloc, AffineMap::get(2, 0, Uexprs, nestedBuilder.getContext()), ivs);
        auto Lloaded = nestedBuilder.create<AffineLoadOp>(loc, target, AffineMap::get(2, 0, Lexprs, nestedBuilder.getContext()), ivs);
        auto Ldiv = nestedBuilder.create<DivFOp>(loc, Lloaded, uzero);
        nestedBuilder.create<AffineStoreOp>(loc, Ldiv, alloc, AffineMap::get(2, 0, Lexprs, nestedBuilder.getContext()), ivs);
    });

    for(int i = 1;i < looplen; i++){
        SmallVector<int64_t, 2> fourthlowerBounds;
        SmallVector<int64_t, 2> fourthupperBounds;
        SmallVector<int64_t, 2> fourthsteps(2, 1);
        fourthlowerBounds.push_back(i);
        fourthlowerBounds.push_back(0);
        fourthupperBounds.push_back(looplen);
        fourthupperBounds.push_back(i);
        buildAffineLoopNest(
          rewriter, loc, fourthlowerBounds, fourthupperBounds, fourthsteps,
          [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        
            SmallVector<AffineExpr, 2> tempexprs, Uexprs, Lexprs;
            toy::LUOpAdaptor LUAdaptor(operands);
            Value target = LUAdaptor.input();

            tempexprs.push_back(getAffineConstantExpr(2*looplen, nestedBuilder.getContext()));
            tempexprs.push_back(getAffineDimExpr(0, nestedBuilder.getContext()));
            Uexprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()) +
                             getAffineConstantExpr(looplen, nestedBuilder.getContext()));
            Uexprs.push_back(getAffineDimExpr(0, nestedBuilder.getContext()));
            Lexprs.push_back(getAffineConstantExpr(i, nestedBuilder.getContext()));
            Lexprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()));

            auto Uloaded = nestedBuilder.create<AffineLoadOp>(loc, alloc, AffineMap::get(2, 0, Uexprs, nestedBuilder.getContext()), ivs);
            auto Lloaded = nestedBuilder.create<AffineLoadOp>(loc, alloc, AffineMap::get(2, 0, Lexprs, nestedBuilder.getContext()), ivs);
            auto muled = nestedBuilder.create<MulFOp>(loc, Uloaded, Lloaded);
            auto temp = nestedBuilder.create<AffineLoadOp>(loc, alloc, AffineMap::get(2, 0, tempexprs, nestedBuilder.getContext()), ivs);
            auto tempadded = nestedBuilder.create<AddFOp>(loc, muled, temp);
            nestedBuilder.create<AffineStoreOp>(loc, tempadded, alloc, AffineMap::get(2, 0, tempexprs, nestedBuilder.getContext()), ivs);
        });
        
        // j iter
        SmallVector<int64_t, 2> fifthlowerBounds;
        SmallVector<int64_t, 2> fifthupperBounds;
        SmallVector<int64_t, 2> fifthsteps(2, 1);
        fifthlowerBounds.push_back(i);
        fifthlowerBounds.push_back(i);
        fifthupperBounds.push_back(i+1);
        fifthupperBounds.push_back(looplen);
        buildAffineLoopNest(
          rewriter, loc, fifthlowerBounds, fifthupperBounds, fifthsteps,
          [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
            SmallVector<AffineExpr, 2> tempexprs, Uexprs, Aexprs;
            toy::LUOpAdaptor LUAdaptor(operands);
            Value target = LUAdaptor.input();

            Uexprs.push_back(getAffineConstantExpr(looplen + i, nestedBuilder.getContext()));
            Uexprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()));
            Aexprs.push_back(getAffineConstantExpr(i, nestedBuilder.getContext()));
            Aexprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()));
            tempexprs.push_back(getAffineConstantExpr(2*looplen, nestedBuilder.getContext()));
            tempexprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()));

            auto Aloaded = nestedBuilder.create<AffineLoadOp>(loc, target, AffineMap::get(2, 0, Aexprs, nestedBuilder.getContext()), ivs);
            auto temp = nestedBuilder.create<AffineLoadOp>(loc, alloc, AffineMap::get(2, 0, tempexprs, nestedBuilder.getContext()), ivs);
            auto subed = nestedBuilder.create<SubFOp>(loc, Aloaded, temp);
            nestedBuilder.create<AffineStoreOp>(loc, subed, alloc, AffineMap::get(2, 0, Uexprs, nestedBuilder.getContext()), ivs);
            const APFloat zero(0.0);
            auto Zero = rewriter.create<ConstantFloatOp>(loc,zero,nestedBuilder.getF64Type());
            nestedBuilder.create<AffineStoreOp>(loc, Zero, alloc, AffineMap::get(2, 0, tempexprs, nestedBuilder.getContext()), ivs); 
        });
        
        SmallVector<int64_t, 2> sixthlowerBounds;
        SmallVector<int64_t, 2> sixthupperBounds;
        SmallVector<int64_t, 2> sixthsteps(2, 1);
        sixthlowerBounds.push_back(i + 1);
        sixthlowerBounds.push_back(0);
        sixthupperBounds.push_back(looplen);
        sixthupperBounds.push_back(i);
        buildAffineLoopNest(
          rewriter, loc, sixthlowerBounds, sixthupperBounds, sixthsteps,
          [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        
            SmallVector<AffineExpr, 2> tempexprs, Uexprs, Lexprs;
            toy::LUOpAdaptor LUAdaptor(operands);
            Value target = LUAdaptor.input();

            tempexprs.push_back(getAffineConstantExpr(2*looplen, nestedBuilder.getContext()));
            tempexprs.push_back(getAffineDimExpr(0, nestedBuilder.getContext()));
            Uexprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()) +
                             getAffineConstantExpr(looplen, nestedBuilder.getContext()));
            Uexprs.push_back(getAffineConstantExpr(i, nestedBuilder.getContext()));
            Lexprs.push_back(getAffineDimExpr(0, nestedBuilder.getContext()));
            Lexprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()));

            auto Uloaded = nestedBuilder.create<AffineLoadOp>(loc, alloc, AffineMap::get(2, 0, Uexprs, nestedBuilder.getContext()), ivs);
            auto Lloaded = nestedBuilder.create<AffineLoadOp>(loc, alloc, AffineMap::get(2, 0, Lexprs, nestedBuilder.getContext()), ivs);
            auto muled = nestedBuilder.create<MulFOp>(loc, Uloaded, Lloaded);
            auto temp = nestedBuilder.create<AffineLoadOp>(loc, alloc, AffineMap::get(2, 0, tempexprs, nestedBuilder.getContext()), ivs);
            auto tempadded = nestedBuilder.create<AddFOp>(loc, muled, temp);
            nestedBuilder.create<AffineStoreOp>(loc, tempadded, alloc, AffineMap::get(2, 0, tempexprs, nestedBuilder.getContext()), ivs);
        });

        // j iter
        SmallVector<int64_t, 2> seventhlowerBounds;
        SmallVector<int64_t, 2> seventhupperBounds;
        SmallVector<int64_t, 2> seventhsteps(2, 1);
        seventhlowerBounds.push_back(i);
        seventhlowerBounds.push_back(i+1);
        seventhupperBounds.push_back(i+1);
        seventhupperBounds.push_back(looplen);
        buildAffineLoopNest(
          rewriter, loc, seventhlowerBounds, seventhupperBounds, seventhsteps,
          [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
            SmallVector<AffineExpr, 2> tempexprs, Uexprs, Aexprs;
            toy::LUOpAdaptor LUAdaptor(operands);
            Value target = LUAdaptor.input();

            Uexprs.push_back(getAffineConstantExpr(looplen + i, nestedBuilder.getContext()));
            Uexprs.push_back(getAffineConstantExpr(i, nestedBuilder.getContext()));
            Aexprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()));
            Aexprs.push_back(getAffineConstantExpr(i, nestedBuilder.getContext())); 
            tempexprs.push_back(getAffineConstantExpr(2*looplen, nestedBuilder.getContext()));
            tempexprs.push_back(getAffineDimExpr(1, nestedBuilder.getContext()));

            auto Aloaded = nestedBuilder.create<AffineLoadOp>(loc, target, AffineMap::get(2, 0, Aexprs, nestedBuilder.getContext()), ivs);
            auto temp = nestedBuilder.create<AffineLoadOp>(loc, alloc, AffineMap::get(2, 0, tempexprs, nestedBuilder.getContext()), ivs);
            auto subed = nestedBuilder.create<SubFOp>(loc, Aloaded, temp);
            auto Uloaded = nestedBuilder.create<AffineLoadOp>(loc, alloc, AffineMap::get(2, 0, Uexprs, nestedBuilder.getContext()), ivs);
            auto dived = nestedBuilder.create<DivFOp>(loc, subed, Uloaded);
            nestedBuilder.create<AffineStoreOp>(loc, dived, alloc, AffineMap::get(2, 0, Aexprs, nestedBuilder.getContext()), ivs);
            const APFloat zero(0.0);
            auto Zero = rewriter.create<ConstantFloatOp>(loc,zero,nestedBuilder.getF64Type());
            nestedBuilder.create<AffineStoreOp>(loc, Zero, alloc, AffineMap::get(2, 0, tempexprs, nestedBuilder.getContext()), ivs); 
        });
    }//end of for
    
    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};


struct LUplusOpLowering : public ConversionPattern {
  LUplusOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::LUplusOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
   
    auto loc = op->getLoc();
    auto resultType = (*op->result_type_begin()).cast<TensorType>();
    auto memRefType = convertTensorToMemRef(resultType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    SmallVector<int64_t, 2> firstlowerBounds(2, /*Value=*/0);
    SmallVector<int64_t, 2> steps(2, /*Value=*/1);

    buildAffineLoopNest(
      rewriter, loc, firstlowerBounds, resultType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {

        toy::LUplusOpAdaptor LUplusAdaptor(operands);
        Value target = LUplusAdaptor.input();
        auto Aloaded = nestedBuilder.create<AffineLoadOp>(loc, target, ivs);
        nestedBuilder.create<AffineStoreOp>(loc, Aloaded, alloc, ivs); 
      });

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

struct DetOpLowering : public ConversionPattern {
  DetOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::DetOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
   
    auto loc = op->getLoc();
    auto resultType = (*op->result_type_begin()).cast<TensorType>();
    auto targetType = op->getOperand(0).getType().cast<TensorType>();
    auto looplen = targetType.getShape().vec()[1];
    auto memRefType = convertTensorToMemRef(resultType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    SmallVector<int64_t, 2> firstlowerBounds(2, /*Value=*/0);
    SmallVector<int64_t, 2> firstupperBounds;
    SmallVector<int64_t, 2> steps(2, /*Value=*/1);
    firstupperBounds.push_back(1);
    firstupperBounds.push_back(1);
    buildAffineLoopNest(
      rewriter, loc, firstlowerBounds, firstupperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        
        const APFloat onee(1.0);
        auto One = rewriter.create<ConstantFloatOp>(loc,onee,nestedBuilder.getF64Type());
        nestedBuilder.create<AffineStoreOp>(loc, One, alloc, ivs); 
      });

    SmallVector<int64_t, 2> secondlowerBounds(2, /*Value=*/0);
    SmallVector<int64_t, 2> secondupperBounds;
    secondupperBounds.push_back(looplen);
    secondupperBounds.push_back(1);

    buildAffineLoopNest(
      rewriter, loc, secondlowerBounds, secondupperBounds, steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        
        SmallVector<AffineExpr, 2> Uexprs, exprs;
        toy::DetOpAdaptor DetAdaptor(operands);
        Value target = DetAdaptor.input();

        Uexprs.push_back(getAffineDimExpr(0, nestedBuilder.getContext()) + 
                         getAffineConstantExpr(looplen, nestedBuilder.getContext()));
        Uexprs.push_back(getAffineDimExpr(0, nestedBuilder.getContext()));
        exprs.push_back(getAffineConstantExpr(0, nestedBuilder.getContext()));
        exprs.push_back(getAffineConstantExpr(0, nestedBuilder.getContext())); 
      
        auto det = nestedBuilder.create<AffineLoadOp>(loc, alloc, AffineMap::get(2, 0, exprs, nestedBuilder.getContext()), ivs);
        auto Uloaded = nestedBuilder.create<AffineLoadOp>(loc, target, AffineMap::get(2, 0, Uexprs, nestedBuilder.getContext()), ivs);
        auto muled = nestedBuilder.create<MulFOp>(loc, det, Uloaded);
        nestedBuilder.create<AffineStoreOp>(loc, muled, alloc, AffineMap::get(2, 0, exprs, nestedBuilder.getContext()), ivs); 
      });

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

} // end anonymous namespace.

//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the toy operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct ToyToAffineLoweringPass
    : public PassWrapper<ToyToAffineLoweringPass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void ToyToAffineLoweringPass::runOnFunction() {
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
  target.addLegalDialect<AffineDialect, StandardOpsDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as `legal`.
  target.addIllegalDialect<toy::ToyDialect>();
  target.addLegalOp<toy::PrintOp>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  OwningRewritePatternList patterns;
  patterns.insert<AddOpLowering, SubOpLowering, ConstantOpLowering, MulOpLowering,
                  ReturnOpLowering, TransposeOpLowering, ConvValidOpLowering, 
                  FillFullOpLowering, FillSomeOpLowering, MatrixMulOpLowering,
                  LUOpLowering, LUplusOpLowering, CmpOpLowering, DetOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(getFunction(), target, patterns)))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}
