# For a better template to copy, see examples/standalone
cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

project(mlir-toy LANGUAGES CXX C)

# Find MLIR package
find_package(MLIR REQUIRED CONFIG)

# Check and find CUDA package
if (NOT ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD))
  message(SEND_ERROR
    "Building the mlir cuda runner requires the NVPTX backend")
endif()

# Configure CUDA runner support. Using check_language first allows us to give
# a custom error message.
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(SEND_ERROR
    "Building the mlir cuda runner requires a working CUDA install")
endif()

# We need the libcuda.so library.
find_library(CUDA_RUNTIME_LIBRARY cuda)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
add_definitions(${LLVM_DEFINITIONS})

include_directories(include)
add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )

set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
mlir_tablegen(ToyCombine.inc -gen-rewriters)
add_public_tablegen_target(ToyCombineIncGen)

set(LLVM_OPTIONAL_SOURCES
  toyc.cpp
  toy_cuda_runner.cpp
  cuda-runtime-wrappers.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/LowerToAffineLoops.cpp
  mlir/LowerToGPU.cpp
  mlir/LowerToLLVM.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp
  cuda-runtime-wrappers.cpp
  )

add_llvm_executable(toyc
  toyc.cpp
  cuda-runtime-wrappers.cpp
  parser/AST.cpp
  mlir/MLIRGen.cpp
  mlir/Dialect.cpp
  mlir/LowerToAffineLoops.cpp
  mlir/LowerToGPU.cpp
  mlir/LowerToLLVM.cpp
  mlir/ShapeInferencePass.cpp
  mlir/ToyCombine.cpp

  DEPENDS
  ToyShapeInferenceInterfaceIncGen
  ToyOpsIncGen
  ToyCombineIncGen
  )


# add_llvm_library(cuda-runtime-wrappers PRIVATE
#   cuda-runtime-wrappers.cpp
# )
# target_include_directories(cuda-runtime-wrappers
#   PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
#   LLVMSupport
# )
# target_link_libraries(cuda-runtime-wrappers
#   PUBLIC
#   LLVMSupport
#   ${CUDA_RUNTIME_LIBRARY}
# )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${CUDA_RUNTIME_LIBRARY}
  LLVMCore
  LLVMSupport
  MLIRJitRunner
  MLIRAnalysis
  MLIREDSC
  MLIRExecutionEngine
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTargetLLVMIR
  MLIRTargetNVVMIR
  MLIRTransforms
  MLIRTranslation
  ${CUDA_RUNTIME_LIBRARY}
)

# Manually expand the target library, since our MLIR libraries
# aren't plugged into the LLVM dependency tracking. If we don't
# do this then we can't insert the CodeGen library after ourselves
llvm_expand_pseudo_components(TARGET_LIBS AllTargetsCodeGens)
# Prepend LLVM in front of every target, this is how the library
# are named with CMake
SET(targets_to_link)
FOREACH(t ${TARGET_LIBS})
  LIST(APPEND targets_to_link "LLVM${t}")
ENDFOREACH(t)

add_llvm_tool(toy-cuda-runner
  toy_cuda_runner.cpp

    DEPENDS
    cuda-runtime-wrappers
    )
  target_include_directories(toy-cuda-runner
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
llvm_update_compile_flags(toy-cuda-runner)
target_link_libraries(toy-cuda-runner PRIVATE ${LIBS} ${targets_to_link})


include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(toyc
  PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    ${LIBS}
    ${targets_to_link}
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRTargetLLVMIR
    MLIRTransforms
    )
