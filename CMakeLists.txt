cmake_minimum_required(VERSION 3.14)

project(splineradon LANGUAGES CXX CUDA)

# set(CMAKE_VERBOSE_MAKEFILE TRUE)

if(NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)


find_package (Python3 COMPONENTS Interpreter Development NumPy)
find_package(xtensor REQUIRED)
find_package(pybind11)
find_package(OpenMP 4.5 REQUIRED)

include_directories(
        # for pybind11
        ${Python3_INCLUDE_DIRS}
        # for xtensor
        ${Python3_INCLUDE_DIRS}/..
        # for xtensor-numpy
        ${Python3_NumPy_INCLUDE_DIRS}
)


set(CMAKE_CXX_STANDARD 14)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -pedantic")
string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
string(APPEND CMAKE_CXX_FLAGS_DEBUG " -O2 -g3")

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60")


add_library(cudaradon SHARED cbi_toolbox/splineradon/src/cudaradon.cpp cbi_toolbox/splineradon/src/tomography_cuda.h cbi_toolbox/splineradon/src/tomography_cuda.cu)
set_target_properties(cudaradon PROPERTIES PREFIX "")

add_library(csplineradon SHARED cbi_toolbox/splineradon/src/csplineradon.cpp cbi_toolbox/splineradon/src/tomography.h cbi_toolbox/splineradon/src/tomography.cpp)
set_target_properties(csplineradon PROPERTIES PREFIX "")

add_library(ompsplineradon SHARED cbi_toolbox/splineradon/src/ompsplineradon.cpp)
set_target_properties(ompsplineradon PROPERTIES PREFIX "")
target_link_libraries(ompsplineradon PUBLIC OpenMP::OpenMP_CXX)
