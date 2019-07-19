cmake_minimum_required(VERSION 3.14)
project(splineradon LANGUAGES CXX CUDA)

find_package (Python3 COMPONENTS Interpreter Development NumPy)

message(${Python3_NumPy_INCLUDE_DIRS})
message(${Python3_LIBRARY_DIRS})

include_directories(
        # for pybind11
        ${Python3_INCLUDE_DIRS}
        # for xtensor
        ${Python3_INCLUDE_DIRS}/..
        # for xtensor-numpy
        ${Python3_NumPy_INCLUDE_DIRS}
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

add_library(cudaradon SHARED cbi_toolbox/splineradon/src/cudaradon.cpp cbi_toolbox/splineradon/src/tomography_cuda.h cbi_toolbox/splineradon/src/tomography_cuda.cu)
set_target_properties(cudaradon PROPERTIES PREFIX "")

add_library(csplineradon SHARED cbi_toolbox/splineradon/src/csplineradon.cpp cbi_toolbox/splineradon/src/tomography.h cbi_toolbox/splineradon/src/tomography.cpp)
set_target_properties(csplineradon PROPERTIES PREFIX "")

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60")