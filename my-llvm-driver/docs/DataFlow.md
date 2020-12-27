## 目录

[TOC]

## 数据流分析实践

本实训项目旨在让你基于LLVM IR学习开发数据流分析Pass。

我们以DCE（dead code elimination）为例，介绍数据流分析及其开发。你需要：

- 回答我们针对示例提出的问题
- 基于[my-llvm-driver/main.cpp](https://gitee.com/s4plus/llvm-ustc-proj/blob/master/my-llvm-driver/main.cpp)和[Driver](https://gitee.com/s4plus/llvm-ustc-proj/blob/master/my-llvm-driver/include/Driver/driver.h#L23)给出的框架代码，自由选择要开展的数据流分析及优化问题 

### 1. 示例代码结构

为了帮助学生尽快熟悉Pass的创建，在[`optimizations/`](../include/optimization)文件夹下提供了[`MyPasses.hpp`](../include/optimization/MyPasses.hpp)文件，其中提供了两个示例Pass的Demo：

+ `myDCEPass`
	+ 继承[`FunctionPass`](https://github.com/llvm/llvm-project/blob/llvmorg-11.0.0/llvm/include/llvm/Pass.h#L284)类
	+ 功能：将转换为SSA格式后的LLVM IR中`use_empty()`返回值为真的指令从指令列表中删除
+ `myGlobalPass`
	+ 继承[`ModulePass`](https://github.com/llvm/llvm-project/blob/llvmorg-11.0.0/llvm/include/llvm/Pass.h#L224)类
	+ 功能：对当前[`Module`](https://github.com/llvm/llvm-project/blob/llvmorg-11.0.0/llvm/include/llvm/IR/Module.h#L67)所定义的函数数目进行计数

### 2. 示例说明及问题

#### 创建一个LLVM Pass

首先确定你要创建的Pass所作用的主体是Module（一个程序文件）还是Function（一个函数），从而决定是从[ModulePass](https://github.com/llvm/llvm-project/blob/llvmorg-11.0.0/llvm/include/llvm/Pass.h#L224)还是从[FunctionPass](https://github.com/llvm/llvm-project/blob/llvmorg-11.0.0/llvm/include/llvm/Pass.h#L284)继承。不论是从哪继承，其创建过程没有太大区别：

```c++
namespace llvm {
    ModulePass * createmyToyPass();
    void initializemyToyPassPass(PassRegistry&);
}

namespace {
    class myToyPass : public ModulePass {
        public:
            static char ID;
            myToyPass() : ModulePass(ID) {
                initializemyToyPassPass(*PassRegistry::getPassRegistry());
            }
            bool runOnModule(Module& M) override {
      			// 如果是继承FunctionPass，这里的runOnModule函数需要替换成
      			// bool runOnFunction(Function& F) override {
            }
    };
}   

char myToyPass::ID = 0;
INITIALIZE_PASS(myToyPass, "mytoy", "My Toy Pass", false, false)
ModulePass *llvm::createmyToyPass() {
    return new myToyPass();
}
```

创建模板如上所示，主要分为两个部分：**Pass的定义**以及**Pass的注册**。

##### 1）Pass的定义

在定义Pass的过程中，需要做以下步骤：

+ 定义一个公共成员变量`ID`
+ 在构造函数中调用`intialize<passname>Pass`接口，该函数在使用`INITIALIZE_PASS`宏时被声明
+ 定义Pass的主体`runOnModule`或者`runOnFunction`函数，在这个函数中实现你需要这个Pass完成的功能

##### 2）Pass的注册

在注册Pass的过程中，需要做如下步骤：

+ 初始化静态成员变量`ID`
+ 声明宏`INITIALIZE_PASS(<passname>, <pass-info-arg>, <pass-info-name>, false, false)`。其中`<passname>`是你的Pass类名，`<pass-info-arg>`可以输出你的Pass的缩写，`<pass-info-name>`可以填入你的Pass的全程。
+ 声明并定义Pass的创建函数`create<passname>`，这一函数可以在往Pass Manager中添加Pass时调用

#### 理解DCE Pass

在[`MyPasses.hpp`](../include/optimization/MyPasses.hpp)中定义了[myDCEPass](../include/optimization/MyPasses.hpp#L69)类，这是一个FunctionPass，它重写了 `bool runOnFunction(Function &F)`。请阅读并实践，回答：
1）简述`skipFunction()`、`getAnalysisIfAvailable<TargetLibraryInfoWrapperPass>()`的功能

2）请简述DCE的数据流分析过程，即`eliminateDeadCode()`和`DCEInstruction()`

#### 理解Global Pass

在[`MyPasses.hpp`](../include/optimization/MyPasses.hpp)中定义了[myGlobalPass](../include/optimization/MyPasses.hpp#L106)类，这是一个ModulePass，它重写了 `bool runOnModule(llvm::Module &M)`。请阅读并实践，回答：
1）简述`skipModule()`的功能

2）请扩展增加对Module中类型定义、全局变量定义等的统计和输出。

### 3. 数据流分析及优化实践

你可以自行选题，开展基于LLVM IR的数据流分析和优化。