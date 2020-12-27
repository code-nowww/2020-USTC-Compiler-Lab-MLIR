# MLIR实践

[MLIR](https://mlir.llvm.org/)（Multi-Level Intermediate Representation）是一种构建可复用、可扩展的编译器基础设施的新方法。MLIR旨在解决软件碎片化，改进对异构硬件的编译，降低构建领域特定编译器的代价，并协助将现有编译器连接在一起。

有关MLIR的演讲及论文参见 [Talks and Related Publications](https://mlir.llvm.org/talks/)，推荐看其中的 CGO 2020 的 keynote。

[MLIR Toy](https://mlir.llvm.org/docs/Tutorials/Toy/)是在MLIR上实现一个玩具语言的教程。它旨在介绍MLIR的概念；特别地，介绍如何用方言 [dialects](https://mlir.llvm.org/docs/LangRef/#dialects) （非正式的说，方言可以理解为不同抽象层次的中间表示）来帮助简化语言构造、程序变换，并向下翻译到LLVM或其他代码生成基础设施。

本项目的[mlir-toy](../)子文件夹包含[MLIR Toy](https://mlir.llvm.org/docs/Tutorials/Toy/)的实现代码，你需要结合[教程](https://mlir.llvm.org/docs/Tutorials/Toy/)来理解代码实现，领悟其中的概念和设计理念，并尝试进行扩展。

简要的说，在代码实现过程中，首先使用递归下降的 parser 解析源代码生成 AST 语法树，在 MLIR 中定义 Toy Dialect 方言和相关算子，然后将 AST 翻译成 Toy Dialect，接着将 Toy Dialect 翻译为 Affine Dialect（需要进行类型和算子的转换），特别地，其中的 print 运算转换成了 LLVM Dialect ，这一部分的实现得益于MLIR能够允许不同的 Dialect 共存。然后再将 Affine Dialect 转换为了 LLVM Dialect，这时整个 module 都是基于 LLVM Dialect，然后可以将整个 LLVM Dialect 转换为 LLVM IR，由LLVM后端进行代码生成。在这一个过程中，需要用户完成的部分是 Toy 语言到 Toy Dialect 的前端解析和翻译，Toy Dialect 的定义和 Toy Dialect 和现有 Dialect 的转换（最终目标是转换为 LLVM Dialect），其他 MLIR 现有 Dialect 到 LLVM IR 转换到代码生成的部分以及对应 Dialect 上的编译优化都已经有对应的实现。

#### 一些可以做的点

- 目前[MLIR Toy](https://mlir.llvm.org/docs/Tutorials/Toy/)支持的算子/类型较少，你可以适当添加一些算子/类型来完成一些有趣的工作
- 在['affine' Dialect](https://mlir.llvm.org/docs/Dialects/Affine/)中，存在一些循环相关的优化，你可以尝试将它们使用起来并进行性能分析
- Toy只使用了MLIR里面的['affine' Dialect](https://mlir.llvm.org/docs/Dialects/Affine/)来进行算子实现，但MLIR里面还有一些高性能的矩阵算法实现，例如，['linalg' Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)。你可以尝试将Toy Dialect转换成['linalg' Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)，以支持开展更多的优化
- MLIR 里面还有很多的[Dialect](https://mlir.llvm.org/docs/Dialects/)，比如['omp' Dialect](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/)和['gpu' Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)，你可以阅读这些Dialect开展更多有趣的尝试
- 除了使用已有 Dialect 的优化，基于不同 Dialect 的抽象表示，你可以选择合适的 Dialect 完成一些自定义的优化

### 其他

- MLIR社区正在开发C / C++/ Fortran 的MLIR Dialect -- [CIL](https://llvm.org/devmtg/2020-09/program/)，但目前还没有开源，所以本次实验很遗憾的不能使用 C 语言作为源语言（预计下一届就能够进行相关实验了），你可以参考相关链接进行进一步了解

愿你的小组在学习实践[MLIR Toy](https://mlir.llvm.org/docs/Tutorials/Toy/)中，不断进行头脑风暴，迸发出想法，打磨优选并实施，你们将有出其不意的收获 :)