cmake_minimum_required(VERSION 3.23)
if(${CMAKE_VERSION} VERSION_LESS 3.23)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(search LANGUAGES CXX)

# Check if we can compile CUDA on this system.
include(CheckLanguage)
check_language(CUDA)

set(CPU_ONLY OFF CACHE BOOL "Build without GPU support?")

if(CMAKE_CUDA_COMPILER AND NOT CPU_ONLY)
  set(HAVE_CUDA 1)
  enable_language(CUDA)
  add_definitions(-DHAVE_CUDA=1)

  # Set the CUDA compile options if we are using a recent CUDA compiler.
  if (NOT CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    set(CMAKE_CUDA17_STANDARD_COMPILE_OPTION "-std=c++17")
    set(CMAKE_CUDA17_EXTENSION_COMPILE_OPTION "-std=c++17")
  endif()
else()
  message(STATUS "Not building for GPU.")
  message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
  message(STATUS "CPU only = ${CPU_ONLY}")
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_subdirectory(lib/pybind11)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    include/
    include/eigen
)


# Create the python module via pybind11.
pybind11_add_module(search MODULE
    src/kbmod/search/bindings.cpp
)

set_target_properties(search PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
if(ipo_supported)
    set_property(TARGET search PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Use aggressive compilation (-O3) and hidden symbol visibility.
target_compile_options(search PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
    -O3
    -fvisibility=hidden
>)

# Link in the OpenMP libraries if they are found.  As of cmake 3.9, it handles setting
# the linking flags and as of cmake 3.12 it works on MacOS (if openmp is installed).
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(search PUBLIC OpenMP::OpenMP_CXX)
    add_definitions(-DHAVE_OPENMP=1)
endif()

# If we have CUDA, build the kernel libraries and link them in as well.
if(HAVE_CUDA)
  message(STATUS "Building CUDA Libraries")
  add_library(searchcu STATIC
      src/kbmod/search/kernels/image_kernels.cu
      src/kbmod/search/kernels/kernel_memory.cu
      src/kbmod/search/kernels/kernels.cu
  )

  set_target_properties(searchcu PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CUDA_VISIBILITY_PRESET "hidden"
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      CUDA_ARCHITECTURES "all"
      PREFIX "${PYTHON_MODULE_PREFIX}"
      SUFFIX "${PYTHON_MODULE_EXTENSION}"
  )
  if(ipo_supported)
      set_property(TARGET searchcu PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()

  target_link_libraries(search PRIVATE searchcu)
else()
  message(STATUS "Skipping CUDA Libraries")
endif()
