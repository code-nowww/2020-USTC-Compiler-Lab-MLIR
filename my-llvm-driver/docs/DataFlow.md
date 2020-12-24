## 文档

本选做实验目标：基于LLVM IR开发数据流分析Pass扩展以DCE为示例，需要给出样例指出问题，需要基于driver给出框架代码，学生可以自由选择优化方向 。

​		为了帮助学生尽快熟悉Pass的创建，在`optimizations/`文件夹下提供了`MyPasses.hpp`文件，其中提供了两个Pass的Demo：

+ `myDCEPass`
	+ 继承`FunctionPass`类
	+ 功能：将转换为SSA格式后的LLVM IR中`use_empty()`返回值为真的指令从指令列表中删除
+ `myGlobalPass`
	+ 继承`ModulePass`类
	+ 功能：对当前`Module`所具有的函数数目进行计数

### 创建一个LLVM Pass

继承FunctionPass或者ModulePass的创建过程没有太大区别：

```c++
namespace llvm {
    FunctionPass * createmyToyPass();
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
						
            }
      			// 如果是继承FunctionPass，这里的runOnModule函数需要替换成
      			// bool runOnFunction(Function& F) override {
            //   
            // }
    };
}   

char myToyPass::ID = 0;
INITIALIZE_PASS(myToyPass, "mytoy", "My Toy Pass", false, false)
FunctionPass *llvm::createmyToyPass() {
    return new myToyPass();
}
```

​		创建模板如上所示，大概可以分为两个部分：Pass的定义以及Pass的注册。

​		在定义Pass的过程中，需要做以下步骤：

+ 定义一个公共成员变量`ID`
+ 在构造函数中调用`intialize<passname>Pass`接口，该函数在使用`INITIALIZE_PASS`宏时被声明
+ 定义Pass的主体`runOnModule`或者`runOnFunction`函数，在这个函数中实现你需要这个Pass完成的功能

​		在注册Pass的过程中，需要做如下步骤：

+ 初始化静态成员变量`ID`
+ 声明宏`INITIALIZE_PASS(<passname>, <pass-info-arg>, <pass-info-name>, false, false)`。其中`<passname>`是你的Pass类名，`<pass-info-arg>`可以输出你的Pass的缩写，`<pass-info-name>`可以填入你的Pass的全程。
+ 声明并定义Pass的创建函数`create<passname>`，这一函数可以在往Pass Manager中添加Pass时调用

