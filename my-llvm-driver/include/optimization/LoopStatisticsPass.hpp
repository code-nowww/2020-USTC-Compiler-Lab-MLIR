#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>
#include <set>

using namespace llvm;

namespace llvm {
    FunctionPass * createLoopStatisticsPass();
    void initializeLoopStatisticsPassPass(PassRegistry&);
}

namespace {

struct LoopStatisticsPass : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid

    LoopStatisticsPass() : FunctionPass(ID) {
        initializeLoopStatisticsPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override {
        if (skipFunction(F))
            return false;
        LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
        std::error_code err;
        raw_fd_ostream outfile_li(StringRef("loop_info.txt"), err);
        LI.print(outfile_li);
        return true;
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.addRequired<LoopInfoWrapperPass>();
        AU.setPreservesCFG();
        AU.addPreserved<LoopInfoWrapperPass>();
        AU.addPreserved<GlobalsAAWrapperPass>();
    }
};

} // end anonymous namespace

char LoopStatisticsPass::ID = 0;
INITIALIZE_PASS(LoopStatisticsPass, "LoopStatisticsPass", "Loop Statistics", false, false)

FunctionPass *llvm::createLoopStatisticsPass() { return new LoopStatisticsPass(); }