# Clang Static Analyzer 实践

在本实训项目中，你将学习使用Clang静态分析器，理解现有的Checker并进行思考，尝试编写自己的Checker。

## 1. 使用Clang Static Analyzer

在深入了解 clang 静态分析的实现机制之前，可以先使用它检测包含有 bug 的 C 程序，初步了解静态分析对程序员到底有何用处。

Clang静态分析可以通过多种渠道来被使用，这里可以使用命令`scan-build`来调用 Clang 静态分析。

例如，如下是一个含有潜在导致悬空引用（全局变量`p`引用了局部变量str的地址）的C程序，假设保存在`test.c`中：

```c++
char *p;
void test()
{
    char str[] = "hello";
    p = str;
}
```

如果用一般的编译器去编译这个程序，可能都不会报任何warning。

接下来可以执行命令：

```
$ scan-build clang -cc1 test.c
```

其中`clang -cc1`指令表示只调用clang的前端进行编译。 你可以看到它会检测出bug，并报了warning。

> 如果具有可视化界面，可以为`scan-build`添加`-v`选项，你还可以看到经过良好排版的网页端结果。

你也可以在直接调用clang命令时指定要使用的某个checker，从而在clang执行期间会调用这个checker对代码进行检查。考虑下面这个程序`testfile.c`。

```c
#include <stdio.h>

FILE *open(char *file)
{
    return fopen(file, "r");
}

void f1(FILE *f)
{
    // do something...
    fclose(f);
}

void f2(FILE *f)
{
    // do something...
    fclose(f);
}

int main()
{
    FILE *f = open("foo");
    f1(f);
    f2(f);
    return 0;
}
```

将这个文件保存于`dblclose.c`，之后执行：

```
$ clang --analyze -Xanalyzer -analyzer-checker=alpha.unix.SimpleStream dblclose.c    
```

即可对这个文件执行SimpleStreamChecker。这个checker的含义如它名字所言。

## 2. 学习现有的checker

Clang 中实现了很多独立的 checker 用来做静态检查。先前看到的`test.c` 中的bug，就是被其中的一个名为(core.StackAddressEscape)的检查器检测出来的。

你可以执行 `clang -cc1 -analyze -analyzer-checker-help` 来查看有哪些checker可用。

所有checker的代码都在[`clang/lib/StaticAnalyzer/Checkers/`](https://github.com/llvm/llvm-project/tree/llvmorg-11.0.0/clang/lib/StaticAnalyzer/Checkers)下。 你可以按照自己的方式阅读。

下面是一些指导和需要回答的问题：

### 2.1 对程序绘制AST、CFG和exploded graph

阅读[这一小节](http://clang-analyzer.llvm.org/checker_dev_manual.html#visualizing)，完成下面几项：

1. 安装 [graphviz](http://www.graphviz.org/)(**注：仅本地安装时可使用**)
2. 写一个含有循环、跳转逻辑的简单程序，保存为`sa/test.c`
3. 使用`clang -cc1 -ast-view test.c`绘制程序的AST，输出保存为`sa/AST.svg`
4. 根据文档说明，绘制`sa/CFG.svg`, `sa/ExplodedGraph.svg`
5. 简要说明`test.c`、`AST.svg`、`CFG.svg`和`ExplodedGraph.svg`之间的联系与区别
6. 特别说明：如果你采用了release配置，或者你无法正常产生svg，你可以选择使用dump选项，并将文字输出放在对应名字的txt中。其他格式的图片也可以接受， 你不需要为格式问题耗费时间。

### 2.2 阅读[Checker Developer Manual](http://clang-analyzer.llvm.org/checker_dev_manual.html)的Static Analyzer Overview一节

回答下面的问题：

1. Checker 对于程序的分析主要在 AST 上还是在 CFG 上进行？
2. Checker 在分析程序时需要记录程序状态，这些状态一般保存在哪里？
3. 简要解释分析器在分析下面程序片段时的过程，在过程中产生了哪些symbolic values? 它们的关系是什么？

一段程序：

```c
int x = 3, y = 4;
int *p = &x;
int z = *(p + 1);
```

### 2.3 简要阅读[LLVM Programmer's Manual](http://llvm.org/releases/11.0.0/docs/ProgrammersManual.html)和[LLVM Coding Standards](http://llvm.org/releases/11.0.0/docs/CodingStandards.html)

这两个manual比较长，你不需要全部阅读，你只需要给出下面几个问题的答案：

1. LLVM 大量使用了 C++11/14的智能指针，请简要描述几种智能指针的特点、使用场合，如有疑问也可以记录在报告中.
2. LLVM 不使用 C++ 的运行时类型推断（RTTI），理由是什么？LLVM 提供了怎样的机制来代替它？
3. 如果你想写一个函数，它的参数既可以是数组，也可以是std::vector，那么你可以声明该参数为什么类型？如果你希望同时接受 C 风格字符串和 std::string 呢？
4. 你有时会在cpp文件中看到匿名命名空间的使用，这是出于什么考虑？

### 2.4 阅读[`clang/lib/StaticAnalyzer/Checkers/SimpleStreamChecker.cpp`](https://github.com/llvm/llvm-project/tree/llvmorg-11.0.0/clang/lib/StaticAnalyzer/Checkers/SimpleStreamChecker.cpp)

回答下面问题：

1. 这个 checker 对于什么对象保存了哪些状态？保存在哪里？
2. 状态在哪些时候会发生变化？
3. 在哪些地方有对状态的检查？
4. 函数`SimpleStreamChecker::checkPointerEscape`的逻辑是怎样的？实现了什么功能？用在什么地方？
5. 根据以上认识，你认为这个简单的checker能够识别出怎样的bug？又有哪些局限性？请给出测试程序及相关的说明。

### 2.5 Checker的编译

阅读[这一节](http://clang-analyzer.llvm.org/checker_dev_manual.html#registration)，以及必要的相关源码，回答下面的问题：

1. 增加一个checker需要增加哪些文件？需要对哪些文件进行修改？
2. 阅读[`clang/include/clang/StaticAnalyzer/Checkers/CMakeLists.txt`](https://github.com/llvm/llvm-project/tree/llvmorg-11.0.0/clang/include/clang/StaticAnalyzer/Checkers/CMakeLists.txt)，解释其中的 clang_tablegen 函数的作用。
3. `.td`文件在clang中出现多次，比如这里的[`clang/include/clang/StaticAnalyzer/Checkers/Checkers.td`](https://github.com/llvm/llvm-project/tree/llvmorg-11.0.0/clang/include/clang/StaticAnalyzer/Checkers/Checkers.td)。这类文件的作用是什么？它是怎样生成C++头文件或源文件的？这个机制有什么好处？

## 3 扩展要求

完成上述基础要求后，你可以考虑完成这部分扩展要求。由于这个扩展要求难度较高，各位请量力而行。

扩展部分提供两种选择，**分析现有 checker 的不足**或者**编写自己的 checker**. 如果觉得很难想到一个有价值的 checker, 建议选择第一个.

### 3.1 分析现有 checker 的缺陷

clang 静态分析提供了对程序中一些常见问题的检查. 比如 cplusplus.NewDelete、unix.Malloc 能对内存泄漏等 bug 进行一定的检查. 但它们并不能解决程序中出现的所有这类问题.

在官网 [Open Project](http://clang-analyzer.llvm.org/open_projects.html) 里提到了一些 clang 静态分析不能解决的问题. 比如:

- **浮点值的处理问题**：当前clang 静态分析器将所有的浮点值处理为`unknown`，故不能静态检测含浮点值的条件，从而审查代码。

```c++
void foo(double f) {
    int *pi = 0;
    if (f > 1.)
        pi = new int;  // bug report: potential leakage
    if (f > 0. && pi)  // must taken, no leakage
        delete pi;
}
```

- **不能处理多次循环**：当前分析器简单地将每个循环展开`N`(比较小的整数) 次

  ```c++
void foo() {
   int *pi = new int;
 for (int i = 0; i < 3; i++) // if replace 3 with 100, no bug report
       if (i == 1000)
           delete pi;          // bug report: potential leakage
  }
  ```
```
  
- **不能处理按位运算**

  ```c
  int main (int argc, char **argv)
  {
    const char *space;
    int flags = argc;
  
    if (flags & (0x01 | 0x02))
        space = "qwe";
  
    if (flags & 0x01)
        return *space; // bug report: Dereference of undefined pointer value
  
    return 0;
  }
```

- 除了这些缺陷以外, clang静态分析器还有哪些缺陷?

- 以动态内存、或文件等资源有关的缺陷检查为例，对clang 静态分析器进行如下使用和分析工作：

  1. 是否能检查该类缺陷?
  2. 检查能力到什么程度（程序存在哪些特征时检查不出来）?
  3. 检查的实现机制是什么？列出相关的源码位置和主要处理流程
  4. （可选）从实现机制上分析，为什么检查不出来上述问题2的解答中所列的特征？
  5. （可选）如果想增强检查能力，可以怎么做？

- 可选的动态内存、或文件等资源有关的缺陷检查

  - cplusplus.NewDelete
  - unix.Malloc
  - unix.API
  - 悬空引用
  - 文件未关闭
  - ......

### 3.2 编写自己的 checker

在阅读clang代码大致了解 checker 的编写方法之后，你可以仔细阅读[Checker Developer Manual](http://clang-analyzer.llvm.org/checker_dev_manual.html)，另外这份[slides](http://llvm.org/devmtg/2012-11/Zaks-Rose-Checker24Hours.pdf) 也是非常好的材料。

作为快速指导，下面将整理一下你需要做的主要工作，和每个阶段工作可能需要参考的材料：

1. 确定你想要检测的bug，或者完成的功能。你可以参考[这个链接](http://clang-analyzer.llvm.org/potential_checkers.html)来寻找一些idea。如果你想要做一种bug checker，你应当先考虑清楚你需要记录哪些状态，状态在哪些时机改变，哪些时机需要对状态进行检查从而确定是否有问题。
2. 开始编码。这里你需要实现你刚刚的想法。你应当已经知道checker应该如何保存状态、如何设置callback、如何获取程序符号，你想知道的其他细节需要在文档中找，另外你会发现你的很多做法会有其他checker可以参考。
3. 注册checker并编译，这里提供两种建议的方式：
   1. 直接通过clang命令行工具，参考[这一节](http://clang-analyzer.llvm.org/checker_dev_manual.html#registration)，核实你的checker可以通过编译，并能够被clang调用。
   2. 通过助教提供的Driver工具注册使用你编写的checker：
      + 在[Checker/myAnalysisAction.cpp](../Checker/myAnalysisAction.cpp)中按照给定的示例注册你编写的checker。
      + 在[main.cpp](../main.cpp)中使用`TheDriver.runChecker()`方法，可以在执行生成的`mClang`时调用你编写的checker。
      + 在[CmakeLists.txt](../CmakeLists.txt)中的`add_executable`函数中加入你编写的`*.cpp`文件。
      + 编译`mClang`工具
4. 编写测试样例进行测试。你的测试样例要能体现出你的checker能完成以及不能完成的事情。注意检查会不会有false positive的情况。测试样例放在`sa/test/`目录下。
5. 编写说明文档。文档中说明你完成的功能，遇到的困难等。

编写好后，你需要提交你的XXXChecker.cpp，以及对你的checker功能的说明（在`README(.md)`中简述）、测试样例。请保证你的程序可以编译通过，之后在实验评测时会将各位的checker统一注册编译。

### 3.3 需要提交的目录格式：

```
sa/
 ├─ README.md               目录文件说明
 ├─ compile.txt             编译记录
 ├─ AST.svg, CFG.svg, ExplodedGraph.svg
 ├─ test.c                  2.1中要求的程序和图
 ├─ answers.txt(或.md)      回答问题
 │                          以上为基础要求部分
 ├─ XXXChecker.cpp          编写的checker源码
 ├─ checker.(md|doc|tex|txt)    对checker的说明
 ├─ analysis.(md|doc|tex|txt)   对 clang 静态分析器的分析
 └─ test/
     ├─ ...                 你的测试样例
```

# 参考

某人写的CSA源码分析的文章, 可以看看 [link](http://blog.csdn.net/dashuniuniu?viewmode=contents), [知乎](https://www.zhihu.com/question/46358643#answer-49189748)
