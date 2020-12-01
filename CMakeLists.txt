# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by François Marelli <francois.marelli@idiap.ch>

# This file is part of CBI Toolbox.

# CBI Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# CBI Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with CBI Toolbox. If not, see <http://www.gnu.org/licenses/>.

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

pybind11_add_module(_cradon cbi_toolbox/splineradon/src/cradon.h cbi_toolbox/splineradon/src/cradon.cpp cbi_toolbox/splineradon/src/ompradon.cpp cbi_toolbox/splineradon/src/cudaradon.cpp)
set_property(TARGET _cradon PROPERTY LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}splineradon/")
target_link_libraries(_cradon PUBLIC OpenMP::OpenMP_CXX)

if(CUDAToolkit_FOUND)
  message(STATUS "CUDA support enabled.")
  add_definitions(-DCUDA)

  target_sources(_cradon PRIVATE cbi_toolbox/splineradon/src/cudaradon.cu)
  set_property(TARGET _cradon PROPERTY CUDA_ARCHITECTURES 60-virtual 61-real)
  set_property(TARGET _cradon PROPERTY LIBRARY_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}splineradon/")  
endif()

