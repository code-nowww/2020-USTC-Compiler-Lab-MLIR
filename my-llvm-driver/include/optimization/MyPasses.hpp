#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/AssumeBundleBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"

#include <iostream>

using namespace llvm;

namespace llvm {
    FunctionPass * createmyDCEPass();
    void initializemyDCEPassPass(PassRegistry&);
}

namespace {
    static bool DCEInstruction(Instruction *I, SmallSetVector<Instruction *, 16> &WorkList, 
        const TargetLibraryInfo * TLI) {
        if (I->use_empty() && !I->isTerminator() && !I->mayHaveSideEffects()) {

            for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
                Value *OpV = I->getOperand(i);
                I->setOperand(i, nullptr);

                if (!OpV->use_empty() || I == OpV)
                    continue;

                if (Instruction *OpI = dyn_cast<Instruction>(OpV))
                    if (isInstructionTriviallyDead(OpI, TLI))
                        WorkList.insert(OpI);
            }
            I->print(llvm::outs());
            std::cout << " " << I->getOpcodeName() << std::endl;
            I->eraseFromParent();
            return true;
        }
        return false;
    }

    static bool eliminateDeadCode(Function &F, TargetLibraryInfo *TLI) {
        bool MadeChange = false;
        SmallSetVector<Instruction *, 16> WorkList;
        std::cout << "The Elimated Instructions: {" << std::endl;
        for (inst_iterator FI = inst_begin(F), FE = inst_end(F); FI != FE;) {
            Instruction *I = &*FI;
            ++FI;

            if (!WorkList.count(I))
            MadeChange |= DCEInstruction(I, WorkList, TLI);
        }

        while (!WorkList.empty()) {
            Instruction *I = WorkList.pop_back_val();
            MadeChange |= DCEInstruction(I, WorkList, TLI);
        }
        std::cout << "}" << std::endl;
        return MadeChange;
    }

    struct myDCEPass : public FunctionPass {
        static char ID;
        myDCEPass() : FunctionPass(ID) {
            initializemyDCEPassPass(*PassRegistry::getPassRegistry());
        }
        bool runOnFunction(Function &F) override {
            if (skipFunction(F)) {
                return false;
            }

            auto * TLIP = getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>();
            TargetLibraryInfo *TLI = TLIP ? &TLIP->getTLI(F) : nullptr;

            return eliminateDeadCode(F, TLI);
        }
        void getAnalysisUsage(AnalysisUsage &AU) const override {
            AU.setPreservesCFG();
        }
    };
}

char myDCEPass::ID = 0;
INITIALIZE_PASS(myDCEPass, "mydce", "My dead code elimination", false, false)
FunctionPass *llvm::createmyDCEPass() {
    return new myDCEPass();
}


//===--------------------------------------------------------------------===//
// Module Pass Demo
//
namespace llvm {
    ModulePass * createmyGlobalPass();
    void initializemyGlobalPassPass(PassRegistry&);
}

namespace {
    class myGlobalPass : public ModulePass {
        public:
            static char ID;
            myGlobalPass() : ModulePass(ID) {
                initializemyGlobalPassPass(*PassRegistry::getPassRegistry());
            }
            bool runOnModule(llvm::Module& M) override {
                if (skipModule(M)) 
                    return false;
                int num_of_func = 0;
                for (llvm::Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++ FI, ++ num_of_func);
                std::cout << "The number of functions is " << num_of_func << "." << std::endl;
                return true;
            }
    };
}   

char myGlobalPass::ID = 0;
INITIALIZE_PASS(myGlobalPass, "global",
                "My Global Pass", false, false)

ModulePass * llvm::createmyGlobalPass() {
    return new myGlobalPass();
}