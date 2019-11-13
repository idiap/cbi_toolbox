cmake_minimum_required(VERSION 3.14)

project(splineradon LANGUAGES CXX CUDA)

# set(CMAKE_VERBOSE_MAKEFILE TRUE)

if(NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

find_package(xtensor-python REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenMP 4.5 REQUIRED)
find_package(Python3 COMPONENTS Interpreter NumPy REQUIRED)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -pedantic")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60")

# pybind11_add_module(cudaradon cbi_toolbox/splineradon/src/cudaradon.cpp cbi_toolbox/splineradon/src/tomography_cuda.h cbi_toolbox/splineradon/src/tomography_cuda.cu)

add_library(cudaradon SHARED cbi_toolbox/splineradon/src/cudaradon.cpp cbi_toolbox/splineradon/src/tomography_cuda.h cbi_toolbox/splineradon/src/tomography_cuda.cu)
set_target_properties(cudaradon PROPERTIES PREFIX "")
set_property(TARGET cudaradon PROPERTY CXX_STANDARD 14)
target_include_directories(cudaradon PRIVATE ${pybind11_INCLUDE_DIRS})

pybind11_add_module(csplineradon cbi_toolbox/splineradon/src/csplineradon.cpp cbi_toolbox/splineradon/src/tomography.h cbi_toolbox/splineradon/src/tomography.cpp)
target_include_directories(csplineradon PRIVATE ${xtensor-python_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})

pybind11_add_module(ompsplineradon cbi_toolbox/splineradon/src/ompsplineradon.cpp)
target_compile_options(ompsplineradon PRIVATE ${OpenMP_CXX_FLAGS})
