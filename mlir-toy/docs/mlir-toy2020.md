# MLIR实践

[MLIR](https://mlir.llvm.org/)（Multi-Level Intermediate Representation）是一种构建可复用、可扩展的编译器基础设施的新方法。MLIR旨在解决软件碎片化，改进对异构硬件的编译，降低构建领域特定编译器的代价，并协助将现有编译器连接在一起。

有关MLIR的演讲及论文参见 https://mlir.llvm.org/talks/。

[MLIR Toy]([Toy Tutorial - MLIR (llvm.org)](https://mlir.llvm.org/docs/Tutorials/Toy/))是在MLIR上实现一个玩具语言的教程。它旨在介绍MLIR的概念；特别地，介绍如何用方言 [dialects](https://mlir.llvm.org/docs/LangRef/#dialects) 来帮助简化语言构造、程序变换，并向下翻译到LLVM或其他代码生成基础设施。

本项目的[mlir-toy](./mlir-toy)子文件夹包含[MLIR Toy]([Toy Tutorial - MLIR (llvm.org)](https://mlir.llvm.org/docs/Tutorials/Toy/)的实现代码，你需要结合[教程](https://mlir.llvm.org/docs/Tutorials/Toy/)来理解代码实现，领悟其中的概念和设计理念，并尝试进行扩展。

#### 一些可以做的点

- 目前[MLIR Toy](https://mlir.llvm.org/docs/Tutorials/Toy/)支持的算子较少，你可以适当添加一些算子来完成一些有趣的工作
- 在['affine' Dialect](https://mlir.llvm.org/docs/Dialects/Affine/)中，存在一些循环相关的优化，你可以尝试将它们使用起来并进行性能分析
- Toy只使用了MLIR里面的['affine' Dialect](https://mlir.llvm.org/docs/Dialects/Affine/)来进行算子实现，但MLIR里面还有一些高性能的矩阵算法实现，例如，['linalg' Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)。你可以尝试将Toy Dialect转换成['linalg' Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)，以支持开展更多的优化
- MLIR 里面还有很多的[Dialect](https://mlir.llvm.org/docs/Dialects/)，比如['omp' Dialect](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/)和['gpu' Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)，你可以阅读这些Dialect开展更多有趣的尝试

愿你的小组在学习实践[MLIR Toy](https://mlir.llvm.org/docs/Tutorials/Toy/)中，不断进行头脑风暴，迸发出想法，打磨优选并实施，你们将有出其不意的收获 :)