# Toy Tutorial

This contains sample code to support the tutorial on using MLIR for building a compiler for a simple Toy language.

This example comes from the [Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) of MLIR, refer to the link for more information. The main code comes from [mlir/examples/toy/Ch7](https://github.com/llvm/llvm-project/tree/llvmorg-11.0.0/mlir/examples/toy/Ch7) of LLVM-11.0.0. The CMakeLists.txt is modified referring to the example of [mlir/examples/standalone](https://github.com/llvm/llvm-project/tree/llvmorg-11.0.0/mlir/examples/standalone) so that it can be compiled out-of-tree.

You can see the [mlir-toy2020.md](./docs/mlir-toy2020.md) for more information and next steps. 

## Building

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX` (see [Getting Started](https://mlir.llvm.org/getting_started/) to compile MLIR). To build this sample, run
```sh
mkdir build && cd build
# if use ninja: https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir
# else
# cmake .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir
cmake --build .
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
The `toyc` compiler is located in `./bin/toyc` and documentation is located in `docs/Toy.md`.

## Runing


```sh
# get help
./bin/toyc --help
# run ../tests/simple.toy
./bin/toyc -emit=jit ../tests/simple.toy
```

You can get other types of output by passing the `-emit` parameter:
```sh
--emit=<value>                                            - Select the kind of output desired
  =ast                                                    -   output the AST dump
  =mlir                                                   -   output the MLIR dump
  =mlir-affine                                            -   output the MLIR dump after affine lowering
  =mlir-llvm                                              -   output the MLIR dump after llvm lowering
  =llvm                                                   -   output the LLVM IR dump
  =jit                                                    -   JIT the code and run it by invoking the main function
```

