cmake_minimum_required(VERSION 3.5)
project(kbmod)
find_package(CUDA REQUIRED)
find_package( PythonInterp 3.6 REQUIRED )
find_package( PythonLibs 3.6 REQUIRED )
add_subdirectory(pybind11)

set(
   CUDA_NVCC_FLAGS
   ${CUDA_NVCC_FLAGS};
   -O3 -gencode arch=compute_60,code=sm_60
   )

set(CUDA_64_BIT_DEVICE_CODE ON)

set(
   CMAKE_CXX_FLAGS "-std=c++11 -O3 -fvisibility=hidden -fopenmp" # -flto
   )

include_directories(
   ../include
   ../src
   ./pybind11/include
   /usr/local/cuda/samples/common/inc
   )

link_directories(
   ../lib
   /usr/local/cuda/samples/common/lib/linux/x86_64
   /usr/local/cuda/lib64
   )

cuda_add_library(
   kbmod 
   MODULE 
   classBindings.cpp
   ../src/kernels.cu
   )

target_link_libraries(
   kbmod
   pybind11::module
   -lcfitsio
   -lgomp
   cudart
   )

set_target_properties(
   kbmod
   PROPERTIES
   PREFIX
   "${PYTHON_MODULE_PREFIX}"
   SUFFIX
   "${PYTHON_MODULE_EXTENSION}"
   )
