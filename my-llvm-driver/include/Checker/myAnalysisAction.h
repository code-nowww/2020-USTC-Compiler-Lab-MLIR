#ifndef __MY_ANALYSIS_ACTION_H__
#define __MY_ANALYSIS_ACTION_H__

#include "clang/Frontend/FrontendAction.h"
#include "clang/StaticAnalyzer/Frontend/FrontendActions.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Core/AnalyzerOptions.h"


#include <vector>

using namespace llvm;
using namespace clang;
using namespace clang::ento;


class myAnalysisAction : public ASTFrontendAction {
  
private:
  template <class checker>
  void addChecker(CompilerInstance &CI, AnalysisASTConsumer * AnalysisConsumer, llvm::StringRef FullName) {
    // 注册自定义的checker
    AnalysisConsumer->AddCheckerRegistrationFn([FullName] (CheckerRegistry& Registry) {
      Registry.addChecker<checker>(FullName,"No desc", "No DocsUri");
    });
    AnalyzerOptionsRef AnalyzerOptions = CI.getAnalyzerOpts();

    // 将自定义的checker加入到AnalyzerOptions当中，使之能够被调用
    AnalyzerOptions->CheckersAndPackages.push_back({FullName.str(), true});
  }

public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(
    CompilerInstance &CI, llvm::StringRef InFile) override;


  
};


#endif