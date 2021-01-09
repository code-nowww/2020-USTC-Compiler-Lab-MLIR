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
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
// #include "llvm/IR/IRBuilder.h"
// #include "llvm/IR/InstIterator.h"
// #include "llvm/IR/InstrTypes.h"
// #include "llvm/IR/Instruction.h"
// #include "llvm/IR/Instructions.h"
// #include "llvm/IR/IntrinsicInst.h"
// #include "llvm/IR/PassManager.h"
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
#include <map>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Dominators.h"

using namespace llvm;
using BB = BasicBlock;

namespace llvm {
  FunctionPass * createLoopStatisticsPass();
  void initializeLoopStatisticsPassPass(PassRegistry&);
}

namespace {
class Loop;
class LoopStat;
void discoverAndMapSubloop(Loop *L, ArrayRef<BB *> Backedges,
                           LoopStat *LI,
                           const DomTreeBase<BB> &DomTree);

class Loop {
private:
  Loop* ParentLoop;
  BB* header;

public:
  Loop(BB* header): header(header) {}
  BB* getHeader() { return header; }
  Loop* getParentLoop() const { return ParentLoop; }
  void setParentLoop(Loop* L) { ParentLoop = L; }

};


class LoopStat {
private:
  std::map<const BasicBlock *, Loop *> BBMap;
  std::vector<Loop*> Loops;

public:
  LoopStat() {}

  void analyze(const DominatorTree &DomTree) {
    // Postorder traversal of the dominator tree.
    auto DomRoot = DomTree.getRootNode();
    for (auto DomNode : post_order(DomRoot)) {

      BasicBlock *Header = DomNode->getBlock();
      std::vector<BasicBlock *> Backedges;

      // Check each predecessor of the potential loop header.
      for (const auto Backedge : children<Inverse<BasicBlock *>>(Header)) {
        // If Header dominates predBB, this is a new loop. Collect the backedges.
        if (DomTree.dominates(Header, Backedge) &&
          DomTree.isReachableFromEntry(Backedge)) {
          Backedges.push_back(Backedge);
        }
      }
      // Perform a backward CFG traversal to discover and map blocks in this loop.
      if (!Backedges.empty()) {
        Loop* L = new Loop(Header);
        Loops.push_back(L);
        discoverAndMapSubloop(L, ArrayRef<BasicBlock *>(Backedges), this, DomTree);
      }
    }
    // Perform a single forward CFG traversal to populate block and subloop
    // vectors for all loops.
    // PopulateLoopsDFS<BasicBlock, Loop> DFS(this);
    // DFS.traverse(DomRoot->getBlock());
  }
  Loop* getLoopFor(BB* block) { return BBMap[block]; }
  void changeLoopFor(BB* block, Loop* L) { BBMap[block] = L; }
};

void discoverAndMapSubloop(Loop *L, ArrayRef<BB *> Backedges,
                            LoopStat *LI,
                            const DomTreeBase<BB> &DomTree) {
  typedef GraphTraits<Inverse<BB *>> InvBlockTraits;

  unsigned NumBlocks = 0;
  unsigned NumSubloops = 0;

  // Perform a backward CFG traversal using a worklist.
  std::vector<BB *> ReverseCFGWorklist(Backedges.begin(), Backedges.end());
  while (!ReverseCFGWorklist.empty()) {
    BB *PredBB = ReverseCFGWorklist.back();
    ReverseCFGWorklist.pop_back();

    Loop *Subloop = LI->getLoopFor(PredBB);
    if (!Subloop) {
    if (!DomTree.isReachableFromEntry(PredBB))
      continue;

      // This is an undiscovered block. Map it to the current loop.
      LI->changeLoopFor(PredBB, L);
      ++NumBlocks;
      if (PredBB == L->getHeader())
        continue;
      // Push all block predecessors on the worklist.
      ReverseCFGWorklist.insert(ReverseCFGWorklist.end(),
                    InvBlockTraits::child_begin(PredBB),
                    InvBlockTraits::child_end(PredBB));
    } else {
      // This is a discovered block. Find its outermost discovered loop.
      while (Loop *Parent = Subloop->getParentLoop())
        Subloop = Parent;

      // If it is already discovered to be a subloop of this loop, continue.
      if (Subloop == L)
        continue;

      // Discover a subloop of this loop.
      Subloop->setParentLoop(L);
      ++NumSubloops;
      PredBB = Subloop->getHeader();
      // Continue traversal along predecessors that are not loop-back edges from
      // within this subloop tree itself. Note that a predecessor may directly
      // reach another subloop that is not yet discovered to be a subloop of
      // this loop, which we must traverse.
      for (const auto Pred : children<Inverse<BB *>>(PredBB)) {
        if (LI->getLoopFor(Pred) != Subloop)
        ReverseCFGWorklist.push_back(Pred);
      }
    }
  }
  // TODO: what is it?
  // L->getSubLoopsVector().reserve(NumSubloops);
  // L->reserveBlocks(NumBlocks);
}

struct LoopStatisticsPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid

  LoopStatisticsPass() : FunctionPass(ID) {
    initializeLoopStatisticsPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    // LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    // std::error_code err;
    // raw_fd_ostream outfile_li(StringRef("loop_info.txt"), err);
    // LI.print(outfile_li);
    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // AU.addRequired<LoopInfoWrapperPass>();
    AU.setPreservesCFG();
    // AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }
};

} // end anonymous namespace

char LoopStatisticsPass::ID = 0;
INITIALIZE_PASS(LoopStatisticsPass, "LoopStatisticsPass", "Loop Statistics", false, false)

FunctionPass *llvm::createLoopStatisticsPass() { return new LoopStatisticsPass(); }