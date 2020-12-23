# MLIR Toy

## 可以做的点

- 目前Toy支持的算子较少，可以适当添加一些算子来完成一些有趣的工作

- 在['affine' Dialect](https://mlir.llvm.org/docs/Dialects/Affine/)中，存在一些循环相关的优化，你可以尝试将它们使用起来并进行性能分析

- Toy只使用了MLIR里面的['affine' Dialect](https://mlir.llvm.org/docs/Dialects/Affine/)来进行算子实现，但MLIR里面还有一些高性能的矩阵算法实现比如['linalg' Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)，可以尝试将Toy Dialect转换成['linalg' Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)进行更多的优化

- MLIR 里面还有很多的[Dialect](https://mlir.llvm.org/docs/Dialects/)，比如['omp' Dialect](https://mlir.llvm.org/docs/Dialects/OpenMPDialect/)和['gpu' Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)，可以阅读这些Dialect以进行更多有趣的尝试