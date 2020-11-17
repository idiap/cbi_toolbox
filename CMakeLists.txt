cmake_minimum_required(VERSION 3.18)

project(splineradon LANGUAGES CXX)

find_package(CUDAToolkit 8.0)

if(CUDAToolkit_FOUND)
  enable_language(CUDA)
else()
  message(WARNING "CUDA toolkit version insufficient, GPU support disabled.")
  message(WARNING "If your CUDAToolkit is not being detected, try setting the environment variable CUDAToolkit_ROOT.")
endif()

# set(CMAKE_VERBOSE_MAKEFILE TRUE)

if(NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(pybind11 REQUIRED)
find_package(OpenMP 4.5 REQUIRED)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -pedantic")

if(CUDAToolkit_FOUND)
  add_library(cudaradon SHARED cbi_toolbox/splineradon/src/cudaradon.cpp cbi_toolbox/splineradon/src/cudaradon.h cbi_toolbox/splineradon/src/cudaradon.cu)

  set_property(TARGET cudaradon PROPERTY CUDA_ARCHITECTURES 60-virtual 61-real)

  set_property(TARGET cudaradon PROPERTY CXX_STANDARD 14)
  target_include_directories(cudaradon PRIVATE ${pybind11_INCLUDE_DIRS})
  target_compile_options(cudaradon PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fvisibility=hidden -std=c++14 -flto -fno-fat-lto-objects>)

  set_target_properties(cudaradon PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
  set_target_properties(cudaradon PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")
  
endif()

pybind11_add_module(ompradon cbi_toolbox/splineradon/src/ompradon.cpp)
target_link_libraries(ompradon PUBLIC OpenMP::OpenMP_CXX)
