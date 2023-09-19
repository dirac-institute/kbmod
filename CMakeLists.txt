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

set(CMAKE_CXX_STANDARD 11)

include_directories(
    include/
    src/kbmod/include/
    src/kbmod/image_base/
)

# --------------------------------------------------
# Build the image_base library and python package --
# --------------------------------------------------

add_library(image_base_lib STATIC
    src/kbmod/image_base/ImageStack.h
    src/kbmod/image_base/ImageStack.cpp
    src/kbmod/image_base/LayeredImage.h
    src/kbmod/image_base/LayeredImage.cpp
    src/kbmod/image_base/PointSpreadFunc.h
    src/kbmod/image_base/PointSpreadFunc.cpp
    src/kbmod/image_base/RawImage.h
    src/kbmod/image_base/RawImage.cpp
)

pybind11_add_module(image_base MODULE
    src/kbmod/image_base/bindings.cpp
)

set_target_properties(image_base PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

if(ipo_supported)
    set_property(TARGET image_base PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET image_base_lib PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

target_compile_options(image_base PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
    -O3
    -fvisibility=hidden
    -fopenmp
>)
target_compile_options(image_base_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
    -O3
    -fvisibility=hidden
    -fopenmp
>)

target_link_libraries(image_base PRIVATE
  ${CFITSIO_LIBRARY}
  -lgomp
)

target_link_libraries(image_base_lib PRIVATE
  ${CFITSIO_LIBRARY}
  -lgomp
)

# If we have CUDA, build the kernel libraries and link them in as well.
if(HAVE_CUDA)
  message(STATUS "Building image_base CUDA Libraries")
  add_library(image_base_cu STATIC
      src/kbmod/image_base/image_kernels.cu
  )

  set_target_properties(image_base_cu PROPERTIES
      POSITION_INDEPENDENT_CODE ON
      CUDA_VISIBILITY_PRESET "hidden"
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
      PREFIX "${PYTHON_MODULE_PREFIX}"
      SUFFIX "${PYTHON_MODULE_EXTENSION}"
  )
  if(ipo_supported)
      set_property(TARGET image_base_cu PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()

  target_link_libraries(image_base PRIVATE image_base_cu)
  target_link_libraries(image_base_lib PRIVATE image_base_cu)
else()
  message(STATUS "Skipping image_base CUDA Libraries")
endif()


# --------------------------------------------------
# Build the search python package ------------------
# --------------------------------------------------

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
  image_base_lib
  ${CFITSIO_LIBRARY}
  -lgomp
)

# If we have CUDA, build the kernel libraries and link them in as well.
if(HAVE_CUDA)
  message(STATUS "Building search CUDA Libraries")
  add_library(searchcu STATIC
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
  message(STATUS "Skipping search CUDA Libraries")
endif()
