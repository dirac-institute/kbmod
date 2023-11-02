cmake_minimum_required(VERSION 3.9...3.17)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
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
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

find_library(CFITSIO_LIBRARY
    NAMES fitsio cfitsio libcfitsio
    HINTS lib/
)

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

target_compile_options(search PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
    -O3
    -fvisibility=hidden
    -fopenmp
>)

target_link_libraries(search PRIVATE
  ${CFITSIO_LIBRARY}
  -lgomp
)


# If we have CUDA, build the kernel libraries and link them in as well.
if(HAVE_CUDA)
  message(STATUS "Building CUDA Libraries")
  add_library(searchcu STATIC
      src/kbmod/search/image_kernels.cu
      src/kbmod/search/kernels.cu
  )

  set_target_properties(searchcu PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CUDA_VISIBILITY_PRESET "hidden"
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
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
