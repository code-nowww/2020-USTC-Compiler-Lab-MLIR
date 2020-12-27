# LLVM Driver Tutorial

This directory contains sample code to support the tutorial on using Clang/LLVM API for building a compiler or analyzer or optimizer for C/C++ or other extended languages.  We define an extensible class `Driver`, which invokes Clang/LLVM API to parse the input program file, to generate AST  and LLVM IR , and then to perform program analysis or transformation on these IRs.

## Directory Structure

```
/
 ├─ CMakeLists.txt          
 ├─ main.cpp                    Program Entry, which calls Driver
 ├─ include
 |   ├─ Checker
 |   |   ├─ myAnalysisAction.h  AST Static Analysis Action
 |   |   └─ SimpleDivZeroChecker.h   Checker Example
 |   ├─ Driver                  Definition of the encapsulated Driver class
 |   └─ optimization
 |       ├─ LoopSearchPass.hpp  Pass for searching Back Edges
 |       └─ MyPasses.hpp        Pass Examples，including Function Pass and Module Pass
 ├─ src                         Definition of functions
 |   ├─ Checker
 |   └─ Driver
 ├─ tests                       Test Cases
 └─ docs                        Documents
     ├─ LoopSearch.md
     ├─ DataFlow.md
     └─ ClangStaticAnalyzer.md
```
## Supported Features
1. Generate the DAG of Clang AST
2. Generate LLVM IR by calling [Clang Driver](https://github.com/llvm/llvm-project/blob/release/11.x/clang/lib/Driver/Driver.cpp), and then  generate the DAG by calling `CFGPrinter`、`DominatorTreePrinter` and further vusualize
3. Optimize IR by calling some LLVM IR TransForm Passes
4. Define class `mDriver::Driver` by encapusulating operations, which support argument parsing, customized `Pass` and printing `LLVM IR` automatically
5. Define class `myAnalysisAction` which invokes  `SimpleDivChecker`  and supports adding new checkers
6. [Driver](src/Driver/driver.cpp) invokes  customized `FunctionPass` and`ModulePass` defined in `optimization/MyPasses.hpp`
7. [Driver](src/Driver/driver.cpp) invokes  `LSPass` defined in `optimization/LoopSearchPass.hpp`, which has already implemented the back edge search algorithm

## Building and Using mDriver::Driver

### 1. Building Clang and LLVM
We have prepared the server with the LLVM Debug version installed for everyone (in-school access is required). The address and account password will be notified separately. If you plan to configure the same experimental environment on your machine, you can refer to [llvm 11.0.0 Installation Documentation](./docs/LLVM-11.0.0-install.md).

### 2. Building and using mDriver::Driver
In the root directory of this software package, execute the following command to compile the driver:
```
mkdir build && cd build
cmake ..
make [-j<num>]
```
The generated executable file is `build/mClang`. The usage method and parameter meaning of mClang are as follows:
```
mClang <source file> [-show-ir-after-pass] [-o=<output path>]
-show-ir-after-pass          print LLVM IR after each Pass invoke
-o=<output path>            specify the output path of LLVM IRs and the default is stdio
-h --help --h               display help menu
```

## Welcome Contributions

The driver provided here still has many shortcomings, such as:

- There may be memory leak problems, and some pointers do not use smart pointer management
- For unverified input codes, there may be exceptions such as segmentation errors
- The backend does not implement assembly code and executable file generation, nor does it implement JIT execution.

Everyone is welcome to find out and ask us questions, and we welcome you to make a pull request to us after solving the problem.

## Contributors

This tutorial software was developed by the [s4plus team](https://s4plus.ustc.edu.cn/) of [Yu Zhang](http://staff.ustc.edu.cn/~yuzhang/) from the University of Science and Technology of China. The developers include Shuo Liu, Yitong Huang, Shunhong Wang, and Mingyu Chen, etc. Teacher Wei Xu is involved in algorithm understanding and environment construction.
