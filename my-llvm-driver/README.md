# LLVM Driver

LLVM Driver的任务是驱动llvm/clang库实现对输入文件的分析，并对产生的AST、LLVM IR等多种中间表示调用LLVM/Clang的API或自己的工具进行解析。

## Driver说明

```
/
 ├─ CMakeLists.txt          
 ├─ main.cpp                    入口程序,调用Driver
 ├─ include
 |   ├─ Checker
 |   |   ├─ myAnalysisAction.h  静态分析Action
 |   |   └─ SimpleDivZeroChecker.h   checker示例
 |   ├─ Driver                  封装的Driver类的定义
 |   └─ optimization
 |       ├─ LoopSearchPass.hpp  查找回边的分析Pass
 |       └─ MyPasses.hpp        示例Pass，函数Pass和模块Pass各一个
 ├─ src                         函数体
 |   ├─ Checker
 |   └─ Driver
 ├─ tests                       测试样例
 └─ docs                        实验说明文档
     ├─ LoopSearch.md
     ├─ DataFlow.md
     └─ ClangStaticAnalyzer.md
```
## Done:
1. 产生了Clang AST的DAG图；
2. 调用[Driver](https://github.com/llvm/llvm-project/blob/release/11.x/clang/lib/Driver/Driver.cpp)产生LLVM IR，
并在这上面调用`CFGPrinter`、`DominatorTreePrinter`产生DAG图，并进行了可视化；
3. 调用LLVM的一些IR TransForm的Pass，对IR进行优化；
4. 封装形成了`mDriver::Driver`类，支持参数解析、定制`Pass`并自动打印`LLVM IR`；
5. `myAnalysisAction`类调用了除0检查，并支持添加新的Checker；
6. Driver调用了`optimization/MyPasses.hpp`中自定义的`FunctionPass`和`ModulePass`；
7. Driver调用了`optimization/LoopSearchPass.hpp`中自定义的`LSPass`，实现程序中的回边查找算法。

## Driver使用说明

### 1. 编译LLVM和clang
我们已经为大家准备了LLVM Debug版本的服务器(需要校内访问)，地址和账号密码在QQ群中已经通知了；如果你打算在本机配置相同的实验环境的话，可以参考[llvm 11.0.0安装文档](./docs/LLVM-11.0.0-install.md)。

### 2. 使用Driver
在driver根目录下，执行下面的命令即可。
```
mkdir build && cd build
cmake ..
make [-j<num>]
```
生成的可执行文件为`build/mClang`，mClang目前可以解析的参数包括：
```
使用方法:mClang <源文件> [-show-ir-after-pass] [-o=<输出IR文件路径>]
-show-ir-after-pass            在每个Pass被调用后打印LLVM IR
-o=<输出IR文件路径>            指定LLVM IRs输出文件路径,若无则输出到标准输出
-h --help --h                  显示帮助菜单
```

## 贡献Driver

Driver目前还存在许多不足的地方，比如：
- 可能存在内存泄露的问题，有些指针没有使用智能指针管理；
- 针对未经验证的输入代码可能会有段错误等异常；
- 后端未实现汇编代码和可执行文件的生成，也未实现JIT执行。

欢迎大家发现并向我们提出问题，欢迎有能力的同学解决问题后向我们提pull request.