## 关于`mlir`使用`cuda`的说明
### 编译
`cmake`时需要指定`-DMLIR_CUDA_RUNNER_ENABLED=ON`和`-DCMAKE_CUDA_COMPILER=nvcc`选项
> PS:如果使用`cuda-9.0`版本时需要指定`-DCMAKE_CUDA_COMPILER=/usr/local/cuda-9.0/bin/nvcc`,另外由于版本兼容性原因,使用大于6.0版本的`gcc`会`cmake`失败，需要手动修改`/usr/local/cuda-9.0/include/crt/host_config.h`第119行,修改`gcc`的版本依赖限制

比如我在集群上使用的编译参数为
```bash
cmake ../llvm -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/opt/llvm-install -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;libcxx;libcxxabi;libunwind;compiler-rt;lld;polly;mlir" -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON  -DMLIR_CUDA_RUNNER_ENABLED=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-9.0/bin/nvcc
```

### 使用`mlir-cuda-runner`
在`llvm/mlir/test/mlir-cuda-runner/`中有关于`mlir cuda`的测试文件，测试文件的运行方式说明如下
```bash
RUN: mlir-cuda-runner %s --shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s
```
`--shared-libs=%cuda_wrapper_library_dir/libcuda-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext`后所跟的是`cuda`的运行时支持,通常来说在`llvm-install`文件夹中。

例如在集群上，运行代码使用的命令如下
```
mlir-cuda-runner all-reduce-and.mlir --shared-libs=/headless/llvm/llvm/llvm-build/lib/libcuda-runtime-wrappers.so,/headless/llvm/llvm/llvm-build/lib/libmlir_runner_utils.so --entry-point-result=void
mlir-cuda-runner all-reduce-and.mlir --shared-libs=/optt/llvm-install/lib/libcuda-runtime-wrappers.so,/optt/llvm-install/lib/libmlir_runner_utils.so --entry-point-result=void
```