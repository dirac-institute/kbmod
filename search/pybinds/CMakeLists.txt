cmake_minimum_required(VERSION 3.5)
project(kbmod)
find_package(CUDA REQUIRED)
find_package( PythonInterp 3.6 REQUIRED )
find_package( PythonLibs 3.6 REQUIRED )
add_subdirectory(pybind11)

# Automatic detection of compute capability from
# https://wagonhelm.github.io/articles/2018-03/detecting-cuda-capability-with-cmake

if (CUDA_FOUND)
  #Get CUDA compute capability
  set(OUTPUTFILE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_script) # No suffix required
  set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/check_cuda.cu)
  execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
  execute_process(COMMAND ${OUTPUTFILE}
                  RESULT_VARIABLE CUDA_RETURN_CODE
                  OUTPUT_VARIABLE ARCH)

  if(${CUDA_RETURN_CODE} EQUAL 0)
    set(CUDA_SUCCESS "TRUE")
  else()
    set(CUDA_SUCCESS "FALSE")
  endif()

  if (${CUDA_SUCCESS})
    message(STATUS "CUDA Architecture: ${ARCH}")
    message(STATUS "CUDA Version: ${CUDA_VERSION_STRING}")
    message(STATUS "CUDA Path: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA Performance Primitives: ${CUDA_npp_LIBRARY}")

    set(CUDA_NVCC_FLAGS "${ARCH}")
    set(CUDA_64_BIT_DEVICE_CODE ON)

    add_definitions(-DGPU) #You may not require this

  else()
    message(WARNING ${ARCH})
  endif()
endif()

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
