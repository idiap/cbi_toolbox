cmake_minimum_required(VERSION 3.18)

project(splineradon LANGUAGES CXX)

find_package(CUDAToolkit 8.0)

if(CUDAToolkit_FOUND)
  enable_language(CUDA)
else()
  message(WARNING "CUDA toolkit version insufficient, GPU support disabled.")
  message(WARNING "If your CUDAToolkit is not being detected, try setting the environment variable CUDAToolkit_ROOT.")
endif()

if(NOT CMAKE_BUILD_TYPE) 
    set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(pybind11)
find_package(OpenMP 4.5 REQUIRED)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -pedantic")

pybind11_add_module(cradon cbi_toolbox/splineradon/src/cradon.h cbi_toolbox/splineradon/src/cradon.cpp cbi_toolbox/splineradon/src/ompradon.cpp cbi_toolbox/splineradon/src/cudaradon.cpp)
set_property(TARGET cradon PROPERTY LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}splineradon/")
target_link_libraries(cradon PUBLIC OpenMP::OpenMP_CXX)

if(CUDAToolkit_FOUND)
  message(STATUS "CUDA support enabled.")
  add_definitions(-DCUDA)

  target_sources(cradon PRIVATE cbi_toolbox/splineradon/src/cudaradon.cu)
  set_property(TARGET cradon PROPERTY CUDA_ARCHITECTURES 60-virtual 61-real)
  set_property(TARGET cradon PROPERTY LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}splineradon/")  
endif()

