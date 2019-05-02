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

    if (Nangles != kernel.shape()[1]) {
        throw py::value_error("The kernel must have Nangle rows.");
    }

    auto sinogram = xt::pytensor<double, 3>({Nangles, Nc, image.shape()[2]});

    radontransform(
            image,      /* Input image */
            h,          /* Sampling step on the image */
            nI,         /* Interpolation degree on the Image */
            x0,         /* Rotation center */
            y0,
            sinogram,   /* Output sinogram of size Nc x Nangles */
            s,          /* Sampling step of the captors */
            nS,         /* Interpolation degree on the Sinogram */
            t0,         /* projection of rotation center*/
            theta,      /* Projection angles in radian */
            kernel,     /* Kernel table of size Nt x Nangles */
            a,          /* Maximal argument of the kernel table (0 to a) */
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

    if (Nangles != kernel.shape()[1]) {
        throw py::value_error("The kernel table must have Nangle columns.");
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
            image,      /* Input image */
            h,          /* Sampling step on the image */
            nI,         /* Interpolation degree on the Image */
            x0,         /* Rotation center */
            y0,
            sinogram,   /* Output sinogram of size Nc x Nangles */
            s,          /* Sampling step of the captors */
            nS,         /* Interpolation degree on the Sinogram */
            t0,         /* projection of rotation center*/
            theta,      /* Projection angles in radian */
            kernel,     /* Kernel table of size Nt x Nangles */
            a,          /* Maximal argument of the kernel table (0 to a) */
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
