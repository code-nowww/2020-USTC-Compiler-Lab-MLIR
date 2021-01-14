
// #include "clang/Frontend/FrontendDiagnostic.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Basic/TargetInfo.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/Utils.h"


#include "Driver/driver.h"
#include "optimization/LoopSearchPass.hpp"
#include "Analysis/LoopStatisticsPass.hpp"
#include "optimization/MyPasses.hpp"

// #include "llvm/Analysis/LoopInfo.h"

using namespace llvm;
using namespace clang;
using namespace mDriver;

// driver封装，main来new driver实例
int main(int argc, const char **argv) {

    // 创建诊断函数
    IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions();
    TextDiagnosticPrinter *DiagClient =
    new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);

    IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());
    DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);

    // 得到当前进程运行所在的目标平台信息，用于JIT中把代码load进当前进程
    const std::string TripleStr = llvm::sys::getProcessTriple();
    llvm::Triple T(TripleStr);

    // 调用Driver，实例化
    llvm::ErrorOr<std::string> ClangPath = llvm::sys::findProgramByName("clang");
    if (!ClangPath) {
        llvm::errs() << "clang not found.\n";
        exit(1);
    }
    llvm::outs() << "Use clang: " <<  ClangPath->c_str() << "\n";
    Driver TheDriver(StringRef(ClangPath->c_str()), T.str(), Diags);
    SmallVector<const char *, 16> Args(argv, argv + argc);
    Args.push_back("-fsyntax-only");
    if(TheDriver.ParseArgs(Args) && TheDriver.BuildCI(Diags)){
        llvm::outs() << "Dump IR successfully.\n";
    }
    else{
        llvm::errs() << "Failed. Early return.\n";
        exit(1);
    }

    TheDriver.FrontendCodeGen();
    TheDriver.runChecker();
    TheDriver.InitializePasses();
    TheDriver.addPass(createPromoteMemoryToRegisterPass());
    TheDriver.addPass(createLSPass());
    TheDriver.addPass(createmyDCEPass());
    TheDriver.addPass(createmyGlobalPass());
    TheDriver.addPass(createLoopStatisticsPass());
    TheDriver.run();
    // TheDriver.printASTUnit();
    // Shutdown.
    llvm::llvm_shutdown();
    return 0;
}
