//
// Created by fmarelli on 16/04/19.
//

#ifndef SRC_SPLINERADON_H
#define SRC_SPLINERADON_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> radon(py::array_t<double> image, double a);

PYBIND11_MODULE(splineradon, m) {
    m.doc() = "Radon transform (and inverse) using spline convolutions discretization";

    m.def("radon", &radon, "Perform radon transform of an image");
}

#endif //SRC_SPLINERADON_H
