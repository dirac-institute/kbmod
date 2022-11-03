cmake_minimum_required(VERSION 3.9...3.17)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(search LANGUAGES CXX CUDA)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)

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
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

if(ipo_supported)
    set_property(TARGET searchcu PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()


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

if(ipo_supported)
    set_property(TARGET search PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()


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
