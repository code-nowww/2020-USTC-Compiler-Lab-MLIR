// #include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Driver/Tool.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/InitializePasses.h"
#include "llvm/Analysis/TargetLibraryInfo.h"

#include "Driver/driver.h"
#include "Checker/myAnalysisAction.h"
#include <iostream>
using namespace clang;
using namespace llvm;
using namespace std;
using namespace clang::ento;

namespace {

// 删除M中的函数中的可能存在的optnone属性
void set_func_attr(llvm::Module *M){
  for(llvm::Module::iterator sF = M->begin(), eF = M->end(); sF != eF; ++sF) {
    // std::cout << "set sF " << sF->getName().str() << " attr to nonopt" << std::endl;
    sF->removeFnAttr(llvm::Attribute::OptimizeNone);
    // sF->removeAttribute(0, llvm::Attribute::OptimizeNone);
  }
}

}// end anonymous namespace

namespace mDriver {

void PrintHelp() {
    llvm::outs() << "中国科大 编译原理H LLVM-Driver-Demo\n";
    llvm::outs() << "使用方法:mClang <源文件> [-show-ir-after-pass] [-o=<输出IR文件路径>]\n";
    llvm::outs() << "-show-ir-after-pass            在每个Pass被调用后打印LLVM IR\n";
    // llvm::outs() << "-dce                           调用死代码删除,Ref:llvm/Transforms/Scalar/DCE.h,默认不调用\n";
    llvm::outs() << "-o=<输出IR文件路径>            指定LLVM IRs输出文件路径,若无则输出到标准输出\n"; 
    llvm::outs() << "-h --help --h                  显示帮助菜单\n";
}

Driver::Driver(StringRef ClangExecutable, StringRef TargetTriple,
         DiagnosticsEngine &Diags) : _TheDriver(ClangExecutable, TargetTriple, Diags)
{
    _TheDriver.setTitle("USTC Compiler Course H2020 Driver");
    llvm::outs() << "Driver \""<< _TheDriver.DriverTitle <<  "\" Created Successfully! \n";
}

bool Driver::ParseArgs(SVec &Args) {
    _show_ir_after_pass = false;
    if(_Args.size()){
        _Args.clear();
    }
    _Args.append(Args.begin(), Args.end());
    for(SVec::iterator s = _Args.begin(), e = _Args.end();s != e; ++s){
        if(0 == strncmp(*s, "-h", 2) || 0 == strncmp(*s, "--help", 6) || 0 == strncmp(*s,"--h", 3)) {
            PrintHelp();
            return false;
        }
        if(0 == strncmp(*s, "-show-ir-after-pass", 19)) {
            _show_ir_after_pass = true;
            _Args.erase(s);
        }
        if(0 == strncmp(*s, "-o=", 3)) {
            _OutFile.clear();
            const char *pos = strstr(*s, "="); pos++;
            _OutFile.append(pos);
            _Args.erase(s);
        }
    }
    // llvm::outs() << "input args size is " << _Args.size() << "\n";
    return true;
}

bool Driver::BuildCI(DiagnosticsEngine &Diags) {
    // 调用Clang Driver根据参数创建编译任务实例
    // Ref:https://github.com/llvm/llvm-project/blob/main/clang/lib/Driver/Driver.cpp#L982
    _C.reset(_TheDriver.BuildCompilation(_Args));
    assert(_C && "Compilation build failed!");
    auto &Jobs = _C->getJobs();
    // 前面创建的编译任务的数目只能为1，i.e. 输入的源代码文件数为1
    // Command: An executable path/name and argument vector to execute.
    if (Jobs.size() != 1 || !isa<driver::Command>(*Jobs.begin())) {
        SmallString<256> Msg;
        llvm::raw_svector_ostream OS(Msg);
        Jobs.Print(OS, "; ", true);
        Diags.Report(diag::err_fe_expected_compiler_job) << OS.str();
        return false;
    }
    auto &Cmd = cast<driver::Command>(*Jobs.begin());
    if (llvm::StringRef(Cmd.getCreator().getName()) != "clang") {
        Diags.Report(diag::err_fe_expected_clang_command);
        return false;
    }
    // 创建一个编译器实例来接收编译任务并设置该实例的诊断器
    std::shared_ptr<CompilerInvocation> CI(new CompilerInvocation);
    CompilerInvocation::CreateFromArgs(*CI, Cmd.getArguments(), Diags);
    _Clang.setInvocation(CI);
    _Clang.createDiagnostics();
    if (!_Clang.hasDiagnostics()) return false;
    return true;
}

bool Driver::FrontendCodeGen(){
    _Act.reset(new EmitLLVMOnlyAction());
    if (!_Clang.ExecuteAction(*_Act))
        return false;
      // LLVM 初始化本地目标程序以及本地LLVM IR的printer函数
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    _M = _Act->takeModule();
    if(!_M) return false;
    if(_OutFile.empty()){
        _M->print(llvm::outs(), nullptr);
    }
    else{
        std::error_code err;
        raw_fd_ostream outfile(StringRef(_OutFile), err);
        _M->print(outfile, nullptr);
    }
    return true;
}

void Driver::InitializePasses(){
    _PassRegistry = PassRegistry::getPassRegistry();
    initializeCore(*_PassRegistry);
    initializeScalarOpts(*_PassRegistry);
    initializeIPO(*_PassRegistry);
    initializeAnalysis(*_PassRegistry);
    initializeTransformUtils(*_PassRegistry);
    initializeTarget(*_PassRegistry);
    if(!_M) {
        llvm::errs() << "mDriver::Driver::_M is nullptr. \n";
        return;
    }
    _PM.reset(new legacy::PassManager());
    const std::string TripleStr = llvm::sys::getProcessTriple();
    llvm::Triple T(TripleStr);
    TargetLibraryInfoImpl TLI(T);
    _PM->add(new TargetLibraryInfoWrapperPass(TLI));
    _FPM.reset(new legacy::FunctionPassManager(_M.get()));
}

bool Driver::runChecker() {
    std::unique_ptr<myAnalysisAction> AAct(new myAnalysisAction());
    bool Res = _Clang.ExecuteAction(*AAct);
    if(Res == true) {
        std::cout << "Execute AnalysisAction Success" << std::endl;
    }
}

void Driver::addPass(FunctionPass *_p){
    _FPM->add(_p);
    if(_show_ir_after_pass){
        auto pass_name = _p->getPassName().str();
        std::replace(pass_name.begin(), pass_name.end(), ' ', '_');
        std::string PassName =  pass_name+ "_" + std::to_string(_PassID++) + ".ll";
        std::error_code _err;
        raw_fd_ostream* outfile = new raw_fd_ostream(StringRef(PassName), _err);
        _FPM->add(_p->createPrinterPass(*outfile, ""));
    }
}

void Driver::addPass(ModulePass *_p){
    _PM->add(_p);
    if(_show_ir_after_pass){
        auto pass_name = _p->getPassName().str();
        std::replace(pass_name.begin(), pass_name.end(), ' ', '_');
        std::string PassName =  pass_name+ "_" + std::to_string(_PassID++) + ".ll";
        std::error_code _err;
        raw_fd_ostream* outfile = new raw_fd_ostream(StringRef(PassName), _err);
        _PM->add(_p->createPrinterPass(*outfile, ""));
    }
}

void Driver::run(){
    if(!_M) {
        llvm::errs() << "mDriver::Driver::_M is nullptr. \n";
        return;
    }
    set_func_attr(_M.get());
    _FPM->doInitialization();
    for(auto sF = _M->begin(), eF = _M->end();sF != eF;++sF){
        _FPM->run(*sF);
    }
    _FPM->doFinalization();
    _PM->run(*_M);
}
}