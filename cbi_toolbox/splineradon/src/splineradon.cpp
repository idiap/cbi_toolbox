//
// Created by fmarelli on 16/04/19.
//

#define FORCE_IMPORT_ARRAY

#include "tomography.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

xt::pytensor<double, 3> radon(
        xt::pytensor<double, 3> &image,
        double h,
        long nI,
        double x0,
        double y0,
        xt::pytensor<double, 1> &theta,
        xt::pytensor<double, 2> &kernel,
        double a,
        const long Nc,
        double s,
        long nS,
        double t0
) {

    if (nI < 0L) {
        throw py::value_error("nI must be greater or equal to 0.");
    }

    if (a < 0.0) {
        throw py::value_error("a, the maximal argument of the lookup table must be a positive.");
    }

    if (Nc < 1L) {
        throw py::value_error("The number of captor must at least be 1.");
    }

    if (nS < -1L) {
        throw py::value_error("nS must be greater of equal to -1.");
    }

    const long Nangles = theta.shape()[0];

    if (Nangles != kernel.shape()[0]) {
        throw py::value_error("The kernel must have Nangle rows.");
    }

    auto sinogram = xt::pytensor<double, 3>({Nangles, Nc, image.shape()[2]});

    radontransform(
            image,
            h,
            nI,
            x0,
            y0,
            sinogram,
            s,
            nS,
            t0,
            theta,
            kernel,
            a,
            false
    );

    return sinogram;
}

xt::pytensor<double, 3> iradon(
        xt::pytensor<double, 3> sinogram,
        double s,
        long nS,
        double t0,
        xt::pytensor<double, 1> theta,
        double h,
        long nI,
        double x0,
        double y0,
        xt::pytensor<double, 2> kernel,
        double a,
        long Nx,
        long Ny
) {

    const long Nangles = sinogram.shape()[0];

    if (nS < 0L) {
        throw py::value_error("nS must be greater or equal to 0.");
    }

    if (Nangles != theta.size()) {
        throw py::value_error("The number of angles in theta in incompatible with the sinogram.");
    }

    if (nI < -1L) {
        throw py::value_error("nI must be greater or equal to -1.");
    }

    if (Nangles != kernel.shape()[0]) {
        throw py::value_error("The kernel table must have Nangle rows.");
    }

    if (a < 0) {
        throw py::value_error("a, the max argument of the lookup table must be positive.");
    }

    if (Nx < 1L) {
        throw py::value_error("Nx must at least be 1.");
    }
    if (Ny < 1L) {
        throw py::value_error("Ny must at least be 1.");
    }

    xt::pytensor<double, 3> image = xt::pytensor<double, 3>({Ny, Nx, sinogram.shape()[2]});

    radontransform(
            image,
            h,
            nI,
            x0,
            y0,
            sinogram,
            s,
            nS,
            t0,
            theta,
            kernel,
            a,
            true
    );

    return image;
}

PYBIND11_MODULE(csplineradon, m) {
    xt::import_numpy();

    m.doc() = "Radon transform (and inverse) using spline convolutions discretization";

    m.def("radon", &radon, "Perform radon transform of an image");
    m.def("iradon", &iradon, "Perform inverse radon transform of a sinogram");
}
