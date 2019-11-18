cmake_minimum_required(VERSION 3.15)

project(splineradon LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
else()
  message(STATUS "No CUDA support, GPU support disabled.")
endif()

# set(CMAKE_VERBOSE_MAKEFILE TRUE)

if(NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find python inside virtual environment first, pass it to pybind11
set(Python3_FIND_STRATEGY LOCATION)
find_package(Python3 COMPONENTS Interpreter NumPy REQUIRED)
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

find_package(pybind11 REQUIRED)
find_package(OpenMP 4.5 REQUIRED)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -pedantic")

if(CMAKE_CUDA_COMPILER)

  string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60")
  
  add_library(cudaradon SHARED cbi_toolbox/splineradon/src/cudaradon.cpp cbi_toolbox/splineradon/src/tomography_cuda.h cbi_toolbox/splineradon/src/tomography_cuda.cu)

  set_property(TARGET cudaradon PROPERTY CXX_STANDARD 14)
  target_include_directories(cudaradon PRIVATE ${pybind11_INCLUDE_DIRS})
  target_compile_options(csplineradon PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fvisibility=hidden -std=c++14 -flto -fno-fat-lto-objects>)

  set_target_properties(cudaradon PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
  set_target_properties(cudaradon PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")  
endif()

pybind11_add_module(ompsplineradon cbi_toolbox/splineradon/src/ompsplineradon.cpp)
target_compile_options(ompsplineradon PRIVATE ${OpenMP_CXX_FLAGS})
