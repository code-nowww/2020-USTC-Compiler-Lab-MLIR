# LLVM驱动程序教程

本目录提供教你用Clang/LLVM应用编程接口构建编译器、程序分析器或优化器的示例代码。我们定义了可扩展的`Driver`类，它调用Clang/LLVM API实现对输入的源程序文件的解析，产生AST、LLVM IR等中间表示，并在这些中间表示上开展程序分析和变换。

## 目录说明

```
/
 ├─ CMakeLists.txt          
 ├─ main.cpp                    入口程序,调用Driver
 ├─ include
 |   ├─ Checker
 |   |   ├─ myAnalysisAction.h  AST静态分析Action
 |   |   └─ SimpleDivZeroChecker.h   checker示例
 |   ├─ Driver                  封装的Driver类的定义
 |   └─ optimization
 |       ├─ LoopSearchPass.hpp  查找回边的分析Pass
 |       └─ MyPasses.hpp        示例Passes，包含函数Pass和模块Pass各一个
 ├─ src                         函数定义
 |   ├─ Checker
 |   └─ Driver
 ├─ tests                       测试样例
 └─ docs                        实验说明文档
     ├─ LoopSearch.md
     ├─ DataFlow.md
     └─ ClangStaticAnalyzer.md
```
## 已支持的功能
1. 产生了Clang AST的DAG图；
2. 调用[Clang Driver](https://github.com/llvm/llvm-project/blob/release/11.x/clang/lib/Driver/Driver.cpp)产生LLVM IR，在此基础上调用`CFGPrinter`、`DominatorTreePrinter`产生DAG图，并进行了可视化；
3. 调用LLVM IR层次的一些TransForm Passes，对IR进行优化；
4. 封装形成了驱动程序框架类`mDriver::Driver`，支持参数解析、定制`Pass`并自动打印`LLVM IR`；
5. 类`myAnalysisAction`调用除0检查，并支持添加新的Checker；
6. Driver类调用了`optimization/MyPasses.hpp`中自定义的`FunctionPass`和`ModulePass`；
7. Driver类调用了`optimization/LoopSearchPass.hpp`中自定义的`LSPass`，实现对给定程序的回边查找算法。

## 使用说明

### 1. 编译Clang和LLVM
我们已经为大家准备了安装有LLVM Debug版本的服务器(需要校内访问)，地址和账号密码另行通知；如果你打算在本机配置相同的实验环境的话，可以参考[llvm 11.0.0安装文档](./docs/LLVM-11.0.0-install.md)。

### 2. 使用Driver
在本软件包根目录下，执行下面的命令编译驱动程序：
```bash
mkdir build && cd build
cmake ..
make [-j<num>]
```
生成的可执行文件为`build/mClang`，mClang的使用方法及参数含义如下：
```bash
mClang <源文件> [-show-ir-after-pass] [-o=<输出IR文件路径>]
```
参数说明：
```bash
-show-ir-after-pass          在每个Pass被调用后打印LLVM IR
-o=<输出IR文件路径>            指定LLVM IRs输出文件路径,若无则输出到标准输出
-h 或 --help 或 --h           显示帮助菜单
```
## 欢迎贡献

这里提供的驱动程序Driver还存在许多不足的地方，比如：
- 可能存在内存泄漏问题，有些指针没有使用智能指针管理；
- 针对未经验证的输入代码可能会有段错误等异常；
- 后端未实现汇编代码和可执行文件的生成，也未实现JIT执行。

欢迎大家发现并向我们提出问题，也欢迎你解决问题后向我们提pull request。

## 研制者

本教程软件由中国科学技术大学[张昱](http://staff.ustc.edu.cn/~yuzhang/)老师[团队](https://s4plus.ustc.edu.cn/)研制，研制者包括刘硕、黄奕桐、王顺洪、陈铭瑜等。徐伟老师参与算法理解和环境搭建等工作。
