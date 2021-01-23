# Introduction to These Codes

by Hank Wang

---

## Files

核心文件树如下:
```tree
mlir-toy/
├── CMakeLists.txt
├── include
│   ├── CMakeLists.txt
│   └── toy
│       ├── AnalysisPass.h
│       ├── AST.h
│       ├── CMakeLists.txt
│       ├── Dialect.h
│       ├── Lexer.h
│       ├── MLIRGen.h
│       ├── Ops.td
│       ├── Parser.h
│       ├── Passes.h
│       ├── ShapeInferenceInterface.h
│       └── ShapeInferenceInterface.td
├── mlir
│   ├── Dialect.cpp
│   ├── LowerToAffineLoops.cpp
│   ├── LowerToLLVM.cpp
│   ├── MLIRGen.cpp
│   ├── ShapeInferencePass.cpp
│   ├── ToyCombine.cpp
│   └── ToyCombine.td
├── parser
│   └── AST.cpp
└── toyc.cpp
```

其中, 
- `Ops.td`: 定义了一些 Operation, 参见[笔记](https://rabbitwhite1.github.io/posts/llvm/2021-1-23-MLIR_ODS.html)
- `ShapeInference*` 对应 [Chapter 4: Enabling Generic Transformation with Interfaces](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/).
- `ToyCombine.*` 对应 [Chapter 3: High-level Language-Specific Analysis and Transformation](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)
- `Dialect.cpp` 定义了 Toy 语言的 dialect 相关信息.


## 增加算子的步骤

### 基于运算符增加算子

1. 修改 Lexer.h, Parser.h 和 MLIRGen.cpp 以支持相应的符号(如减号 `-`, 矩阵乘号 `@`)
2. 修改 Ops.td 增加相应操作, 如 `SubOp`, 此时其实就已经可以生成相应的头文件了.
3. 修改 Dialect.cpp 以实现一些操作. 此处有特殊定制的 `build` 和 `inferShape` 函数(用于形状推断, 是 `ShapeInference` 接口所要求实现的). 应当注意, 如果后面我们有什么其他的更新, 这些算子也都要进行相应的更新.
4. 在 LowerToAffineLoops.cpp 中对此 Op 进行 lowering. 由于已经有相应的支持(`BinaryOpLowering`), 只需要实例化它就好
    ```cpp
    using SubOpLowering = BinaryOpLowering<toy::SubOp, SubFOp>;
    ```
    实例化后, 向 `ToyToAffineLoweringPass::runOnFunction()` 里的 `pattern` 添加此 Lowering(即 `SubOpLowering`)
5. 编译
    如果在集群上编译需要运行
    ```shell
    cmake -G Ninja .. -DMLIR_DIR=/home/ustc/llvm-project-11.0.0/build/lib/cmake/mlir -DLLVM_DIR=/home/ustc/llvm-project-11.0.0/build/lib/cmake/llvm && cmake --build .
    ```
    我在自己电脑上编译了 MLIR 并安装好, 所以只需要:
    ```shell
    mkdir build && cd build
    cmake .. && make
6. 运行测试
    ```
    ./bin/toyc ../tests/subtract.toy -emit=mlir  # 生成 mlir
    ./bin/toyc ../tests/subtract.toy -emit=mlir-affine  # 生成 mlir 和 affine 的混合中间表示
    ./bin/toyc ../tests/subtract.toy -emit=mlir-llvm  # 生成 mlir 和 llvm 的混合中间表示
    ./bin/toyc ../tests/subtract.toy -emit=llvm  # 生成 llvm
    ./bin/toyc ../tests/subtract.toy -emit=jit  # 运行程序
    ```

### 基于内置函数增加算子

TODO