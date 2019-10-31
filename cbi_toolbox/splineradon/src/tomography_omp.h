#ifndef TOMOGRAPHY_OMP_H
#define TOMOGRAPHY_OMP_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

extern void radontransform(
        py::array_t<double, py::array::c_style> &image,      /* Image */
        double h,                            /* Sampling step on the image (pixel size) */
        long nI,                             /* Interpolation degree on the Image */
        double x0,                           /* Rotation center in image coordinates */
        double y0,
        py::array_t<double, py::array::c_style> &sinogram,   /* Sinogram of size Nangles x Nc x Nlayers*/
        double s,                            /* Sampling step of the captors (sinogram "pixel size") */
        long nS,                             /* Interpolation degree on the sinogram */
        double t0,                           /* Projection of rotation center */
        py::array_t<double, py::array::c_style> &theta,      /* Projection angles in radian */
        py::array_t<double, py::array::c_style> &kernel,     /* Kernel table of size Nangles x Nt */
        double a,                            /* Maximal argument of the kernel table (0 to a) */
        bool backprojection                  /* Perform a back-projection */
);

#endif //TOMOGRAPHY_OMP_H
