//
// Created by fmarelli on 05/07/19.
//

#ifndef CUDARADON_H
#define CUDARADON_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

py::array_t<double, py::array::c_style> radon_cuda(
        py::array_t<double, py::array::c_style> &image,
        double h,
        long nI,
        double x0,
        double y0,
        py::array_t<double, py::array::c_style> &theta,
        py::array_t<double, py::array::c_style> &kernel,
        double a,
        const long Nc,
        double s,
        long nS,
        double t0
);

py::array_t<double, py::array::c_style> iradon_cuda(
        py::array_t<double, py::array::c_style> &sinogram,
        double s,
        long nS,
        double t0,
        py::array_t<double, py::array::c_style> &theta,
        py::array_t<double, py::array::c_style> &kernel,
        double a,
        long Nx,
        long Ny,
        double h,
        long nI,
        double x0,
        double y0
);

std::vector<int> compatible_cuda_devices();


#endif //CUDARADON_H
