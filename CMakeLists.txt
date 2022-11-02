cmake_minimum_required(VERSION 3.5)
project(search LANGUAGES CXX CUDA)

#find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package( PythonInterp )
find_package( PythonLibs )

find_library(CFITSIO_LIBRARY 
    NAMES fitsio cfitsio libcfitsio
    HINTS lib/
)

add_subdirectory(lib/pybind11)

set(CMAKE_CXX_STANDARD 11) # set(PYBIND11_CPP_STANDARD -std=c++11)

include_directories(
    include/
)

add_library(searchcu STATIC
    src/kbmod/search/image_kernels.cu
    src/kbmod/search/kernels.cu
)

set_target_properties(searchcu PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
)


pybind11_add_module(search MODULE
    src/kbmod/search/bindings.cpp
)

set_target_properties(search PROPERTIES
    OUTPUT_NAME kbmod/search
    CXX_VISIBILITY_PRESET "hidden"
    INTERPROCEDURAL_OPTIMIZATION TRUE
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_compile_options(search PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
    -O3
    -fvisibility=hidden
    -fopenmp
>)

target_link_libraries(search PRIVATE
    searchcu
    ${CFITSIO_LIBRARY}
    -lgomp
)
