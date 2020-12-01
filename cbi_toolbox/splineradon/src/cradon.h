// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by François Marelli <francois.marelli@idiap.ch>

// This file is part of CBI Toolbox.

// CBI Toolbox is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3 as
// published by the Free Software Foundation.

// CBI Toolbox is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with CBI Toolbox. If not, see <http://www.gnu.org/licenses/>.

#ifndef CRADON_H
#define CRADON_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double, py::array::c_style> radon(
    py::array_t<double, py::array::c_style> &image,
    double h,
    long nI,
    double z0,
    double x0,
    py::array_t<double, py::array::c_style> &theta,
    py::array_t<double, py::array::c_style> &kernel,
    double a,
    const long Nc,
    double s,
    long nS,
    double t0,
    bool use_cuda);

py::array_t<double, py::array::c_style> iradon(
    py::array_t<double, py::array::c_style> &sinogram,
    double s,
    long nS,
    double t0,
    py::array_t<double, py::array::c_style> &theta,
    py::array_t<double, py::array::c_style> &kernel,
    double a,
    long Nz,
    long Nx,
    double h,
    long nI,
    double z0,
    double x0,
    bool use_cuda);

py::array_t<double, py::array::c_style> radon_omp(
    py::array_t<double, py::array::c_style> &image,
    double h,
    long nI,
    double z0,
    double x0,
    py::array_t<double, py::array::c_style> &theta,
    py::array_t<double, py::array::c_style> &kernel,
    double a,
    const long Nc,
    double s,
    long nS,
    double t0);

py::array_t<double, py::array::c_style> iradon_omp(
    py::array_t<double, py::array::c_style> &sinogram,
    double s,
    long nS,
    double t0,
    py::array_t<double, py::array::c_style> &theta,
    py::array_t<double, py::array::c_style> &kernel,
    double a,
    long Nz,
    long Nx,
    double h,
    long nI,
    double z0,
    double x0);

#ifdef CUDA
py::array_t<double, py::array::c_style> radon_cuda(
    py::array_t<double, py::array::c_style> &image,
    double h,
    long nI,
    double z0,
    double x0,
    py::array_t<double, py::array::c_style> &theta,
    py::array_t<double, py::array::c_style> &kernel,
    double a,
    const long Nc,
    double s,
    long nS,
    double t0);

py::array_t<double, py::array::c_style> iradon_cuda(
    py::array_t<double, py::array::c_style> &sinogram,
    double s,
    long nS,
    double t0,
    py::array_t<double, py::array::c_style> &theta,
    py::array_t<double, py::array::c_style> &kernel,
    double a,
    long Nz,
    long Nx,
    double h,
    long nI,
    double z0,
    double x0);

std::vector<int> compatible_cuda_devices();
#endif //CUDA

bool is_cuda_available();

#endif //CRADON_H