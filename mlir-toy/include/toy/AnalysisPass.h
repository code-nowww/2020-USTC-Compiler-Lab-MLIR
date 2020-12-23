#ifndef MLIR_TUTORIAL_TOY_ANALYSISPASS_H_
#define MLIR_TUTORIAL_TOY_ANALYSISPASS_H_
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace toy {

class Analysis {
  public:
    Analysis(Operation *op);
};

class AnalysisPass
    : public PassWrapper<AnalysisPass, OperationPass<ModuleOp>> {
  public:
  AnalysisPass() = default;

  void runOnOperation() override {
    Analysis &myAnalysis = getAnalysis<Analysis>();
    // do something
  }
};

/// Create a Analysis pass.
std::unique_ptr<Pass> createAnalysisPass() {
  return std::make_unique<AnalysisPass>();
}

} // end namespace toy
} // end namespace mlir

#endif // MLIR_TUTORIAL_TOY_ANALYSISPASS_H_