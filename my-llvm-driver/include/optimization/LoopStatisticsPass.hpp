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
#include <string>

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

using LoopStatPtr = std::shared_ptr<LoopStat>;
using LoopPtr = std::shared_ptr<Loop>;

void discoverAndMapSubloop(LoopPtr L, ArrayRef<BB *> Backedges,
                           LoopStat* LI,
                           const DomTreeBase<BB> &DomTree);

class Loop {
private:
  std::string Label;
  LoopPtr ParentLoop;
  BB* Header;
  std::vector<LoopPtr> SubLoops;

public:
  Loop(BB* Header, std::string Label): Header(Header), Label(Label), ParentLoop(nullptr) {}
  BB* getHeader() { return Header; }
  LoopPtr getParentLoop() const { return ParentLoop; }
  void setParentLoop(LoopPtr L) { ParentLoop = L; }
  std::vector<LoopPtr> getSubLoops() { return SubLoops; }
  void addSubLoop(LoopPtr loop) { SubLoops.push_back(loop); }
  std::string getLabel() { return Label; }
  void setLabel(std::string Label) { this->Label = Label; }
  void reverseSubLoops() { std::reverse(SubLoops.begin(), SubLoops.end()); }

  void* relabelAndReorderLoop(std::string Label) {
    setLabel(Label);
    // reverseSubLoops();

    size_t SubLoopSize = SubLoops.size();
    for (size_t i = 0; i < SubLoopSize; i++) {
      SubLoops[i]->relabelAndReorderLoop(Label + std::to_string(i + 1));
    }
  }

};


class LoopStat {
private:
  size_t LoopCounter = 0;
  std::map<const BasicBlock *, LoopPtr> BBMap;
  std::vector<LoopPtr> Loops;
  std::vector<LoopPtr> TopLevelLoops;

public:
  LoopStat() {}

 LoopPtr allocateLoop(BB* Header) {
    std::string Label = "Loop" + std::to_string(LoopCounter++);
    return LoopPtr(new Loop(Header, Label));
  }

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
        LoopPtr L = allocateLoop(Header);
        Loops.push_back(L);
        discoverAndMapSubloop(L, ArrayRef<BasicBlock *>(Backedges), this, DomTree);
      }
    }
    // make the nested loops tree.
    std::set<LoopPtr> DoneSet;
    std::reverse(Loops.begin(), Loops.end());
    for (LoopPtr loop: Loops) {
      if (!loop->getParentLoop()) {
        TopLevelLoops.push_back(loop);
        continue;
      }
      LoopPtr parent = nullptr;
      while (parent = loop->getParentLoop()) {
        // TODO: use find might be more efficient
        if (DoneSet.count(loop) > 0) break;
        parent->addSubLoop(loop);
        DoneSet.insert(loop);
        loop = parent;
      }
    }
    // reverse the loops and label them according to TA's requirements
    size_t TopLevelLoopsSize = TopLevelLoops.size();
    for (size_t i = 0; i < TopLevelLoopsSize; i++) {
      TopLevelLoops[i]->relabelAndReorderLoop("L" + std::to_string(i + 1));
    }
    
  }

  LoopPtr getLoopFor(BB* block) { 
    auto loop = BBMap.find(block);
    if (loop == BBMap.end()) return nullptr;
    else return loop->second;
  }

  void changeLoopFor(BB* block, LoopPtr L) { BBMap[block] = L; }

  void printBase(raw_ostream &OS, LoopPtr loop, size_t Indent) const {
    std::string IndentStr = "";
    for(size_t i = 0; i < Indent; i++) {
      IndentStr += "\t";
    }
    OS << IndentStr << "\"" << loop->getLabel() << "\": {\n";
      OS << IndentStr << "\t\"depth\": " << Indent - 1 << "\n";
      for (auto SubLoop: loop->getSubLoops()) {
        printBase(OS, SubLoop, Indent + 1);
      }
    OS << IndentStr << "}\n";
  }

  void print(raw_ostream &OS, size_t Indent) const {
    for (auto loop: TopLevelLoops) {
      printBase(OS, loop, Indent);
    }
  }

};

void discoverAndMapSubloop(LoopPtr L, ArrayRef<BB *> Backedges,
                           LoopStat* LS,
                           const DomTreeBase<BB> &DomTree) {
  typedef GraphTraits<Inverse<BB *>> InvBlockTraits;

  unsigned NumBlocks = 0;
  unsigned NumSubloops = 0;

  // Perform a backward CFG traversal using a worklist.
  std::vector<BB *> ReverseCFGWorklist(Backedges.begin(), Backedges.end());
  while (!ReverseCFGWorklist.empty()) {
    BB *PredBB = ReverseCFGWorklist.back();
    ReverseCFGWorklist.pop_back();

    LoopPtr Subloop = LS->getLoopFor(PredBB);
    if (!Subloop) {
      if (!DomTree.isReachableFromEntry(PredBB))
        continue;

        // This is an undiscovered block. Map it to the current loop.
        LS->changeLoopFor(PredBB, L);
        ++NumBlocks;
        if (PredBB == L->getHeader())
          continue;
        // Push all block predecessors on the worklist.
        ReverseCFGWorklist.insert(ReverseCFGWorklist.end(),
                      InvBlockTraits::child_begin(PredBB),
                      InvBlockTraits::child_end(PredBB));
    } else {
      // This is a discovered block. Find its outermost discovered loop.
      while (LoopPtr Parent = Subloop->getParentLoop())
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
        if (LS->getLoopFor(Pred) != Subloop)
        ReverseCFGWorklist.push_back(Pred);
      }
    }
  }

}

struct LoopStatisticsPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  LoopStatPtr LS;

  LoopStatisticsPass() : FunctionPass(ID) {
    initializeLoopStatisticsPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    DominatorTree DT(F);
    LS.reset(new LoopStat());
    LS->analyze(DT);
    
    std::error_code err;
    raw_fd_ostream outfile_ls(StringRef(F.getName().str() + "_ls.txt"), err);
    print(outfile_ls, &F);
    
    return true;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // AU.addRequired<LoopInfoWrapperPass>();
    AU.setPreservesCFG();
    // AU.addPreserved<LoopInfoWrapperPass>();
    AU.addPreserved<GlobalsAAWrapperPass>();
  }

  void print(raw_ostream &OS, const Function *F) const {
    OS << "{\n";
      OS << "\t\"" << F->getName().str() << "\": {\n";
        LS->print(OS, 2);
      OS << "\t}\n\n";
    OS << "}";
  }
};

} // end anonymous namespace

char LoopStatisticsPass::ID = 0;
INITIALIZE_PASS(LoopStatisticsPass, "LoopStatisticsPass", "Loop Statistics", false, false)

FunctionPass *llvm::createLoopStatisticsPass() { return new LoopStatisticsPass(); }