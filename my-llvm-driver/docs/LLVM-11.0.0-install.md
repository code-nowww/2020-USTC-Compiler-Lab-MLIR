## 1. 安装LLVM
### 1.1 **注意**
* 用到的LLVM版本限制为`11.0.0`，这是考虑到稳定性以及各版本之间API的细微差别而采取的限制  
* 这里以`Ubuntu 18.04`为例介绍，我们推荐大家(尤其是不太熟悉Linux的同学)采用这个系统。如果你实在想采用其他系统，请自行准备好环境，但是在实验提交中应避免对其他Linux/MacOS发行版环境的依赖
### 1.2 下载LLVM 11.0.0源码并编译
* Step 1. 安装一些必要的依赖
  ``` bash
  sudo apt-get install -y cmake xz-utils build-essential wget
  ```
* Step 2.  请从LLVM网站或其github网站下载源码，选择好你的工作目录，进行解压, 最后得到名为`llvm`的目录
  ``` bash
  # 下载
  github下载源码安装包
  # 解压缩
  tar xvf llvm-11.0.0.src.tar.xz
  mv llvm-11.0.0.src llvm
  tar xvf clang-11.0.0.src.tar.xz
  mv clang-11.0.0.src llvm/tools/clang
  ```
* Step 3. 编译并安装LLVM。这里在内存及硬盘充足的条件下，推荐`Debug`配置的编译，这更能让你体验"较大的项目"的编译过程；否则请采用`Release`配置的编译
  ``` bash
  mkdir llvm-build && cd llvm-build
  # Release
  cmake ../llvm -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=`cd .. && pwd`/llvm-install
  # Debug
  cmake ../llvm -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=`cd .. && pwd`/llvm-install
  # 编译安装,下面的N表示的是你采取的同时执行的编译任务的数目.你需要将其替换为具体的数值,如8,4,1
  # 如果内存充足,一般推荐将N设置为cpu核数的2倍,如果未指明N,将会尽可能多地并行编译
  make -jN install
  # 这一过程可能有半小时至两小时不等,如果内存不足,可以尝试减小N并重新make,已经编译好的部分不会重新编译
  ```
* Step 4.  配置PATH, 使得能方便使用生成的二进制文件。配置PATH在以后的工作中也会是经常用到的，希望大家熟练掌握(或者至少熟练如何搜索)
  ``` bash
  # 将llvm-install/bin目录的完整路径,加入到环境变量PATH中
  # 假设该完整路径为the_path_to_your_llvm-install_bin,
  # 如果你的默认shell是zsh,可以在~/.zshrc中加入一行:
  export PATH=the_path_to_your_llvm-install_bin:$PATH
  # 然后执行
  source ~/.zshrc
  # 如果你的默认shell是bash,欢迎自行google/baidu

  # 如果你的默认shell是bash,可以在~/.bashrc中加入一行:
  cd ~
  vi .bashrc
  ------------在底部增加----------
  export  C_INCLUDE_PATH=the_path_to_your_llvm-install_bin/include:$C_INCLUDE_PATH
  export  CPLUS_INCLUDE_PATH=the_path_to_your_llvm-install_bin/include:$CPLUS_INCLUDE_PATH
  export  PATH=the_path_to_your_llvm-install_bin/bin:$PATH
  -------------------------------
  source .bashrc  
  关闭并再次打开终端
  
  # 最后,检查PATH配置是否成功,执行命令:
  llvm-config --version
  # 成功标志:
  11.0.0 #或类似
  # 失败标志:
  zsh: command not found: llvm-config #或类似
  
  ```

