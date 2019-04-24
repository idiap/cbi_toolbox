//
// Created by fmarelli on 16/04/19.
//

#include "tomography.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<double> radon(
        py::array_t<double> image,
        double h,
        long nI,
        double x0,
        double y0,
        py::array_t<double> theta,
        py::array_t<double> kernel,
        double a,
        long Nc,
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

    py::buffer_info image_info = image.request();
    long Nx = image_info.shape[0];
    long Ny = image_info.shape[1];
    auto Input = (double *) image_info.ptr;

    py::buffer_info theta_info = theta.request();
    if (theta_info.ndim != 1) {
        throw py::value_error("theta must be a 1D array.");
    }
    long Nangles = theta_info.size;
    auto Theta = (double *) theta_info.ptr;

    py::buffer_info kernel_info = kernel.request();
    long Nt = kernel_info.shape[0];
    auto Kernel = (double *) kernel_info.ptr;

    if (Nangles != kernel_info.shape[1]) {
        throw py::value_error("The kernel must have Nangle rows.");
    }

    auto sinogram = py::array_t<double>({Nc, Nangles});
    py::buffer_info sinogram_info = sinogram.request();
    auto Sinogram = (double *) sinogram_info.ptr;


    radontransform(
            Input,      /* Input image */
            Nx,         /* Size of image */
            Ny,
            h,          /* Sampling step on the image */
            nI,         /* Interpolation degree on the Image */
            x0,         /* Rotation center */
            y0,
            Theta,      /* Projection angles in radian */
            Nangles,    /* Number of projection angles */
            Kernel,     /* Kernel table of size Nt x Nangles */
            Nt,         /* Number of samples in the kernel table*/
            a,          /* Maximal argument of the kernel table (0 to a) */
            Sinogram,   /* Output sinogram of size Nc x Nangles */
            Nc,         /* Number of captors */
            s,          /* Sampling step of the captors */
            nS,         /* Interpolation degree on the Sinogram */
            t0          /* projection of rotation center*/
    );

    return sinogram;
}

py::array_t<double> iradon(
        py::array_t<double> sinogram,
        double s,
        long nS,
        double t0,
        py::array_t<double> theta,
        double h,
        long nI,
        double x0,
        double y0,
        py::array_t<double> kernel,
        double a,
        long Nx,
        long Ny
) {
    py::buffer_info sinogram_info = sinogram.request();
    auto Sinogram = (double *) sinogram_info.ptr;
    long Nc = sinogram_info.shape[0];
    long Nangles = sinogram_info.shape[1];

    if (nS < 0L) {
        throw py::value_error("nS must be greater or equal to 0.");
    }

    py::buffer_info theta_info = theta.request();
    auto Theta = (double *) theta_info.ptr;

    if (Nangles != theta_info.shape[0] * theta_info.shape[1]) {
        throw py::value_error("The number of angles in theta in incompatible with the sinogram.");
    }

    if (nI < -1L) {
        throw py::value_error("nI must be greater or equal to -1.");
    }

    py::buffer_info kernel_info = kernel.request();
    auto Kernel = (double *) kernel_info.ptr;

    if (Nangles != kernel_info.shape[1]) {
        throw py::value_error("The kernel table must have Nangle columns.");
    }
    long Nt = kernel_info.shape[0];

    if (a < 0) {
        throw py::value_error("a, the max argument of the lookup table must be positive.");
    }

    if (Nx < 1L){
        throw py::value_error("Nx must at least be 1.");
    }
    if (Ny < 1L){
        throw py::value_error("Ny must at least be 1.");
    }

    auto image = py::array_t<double>({Ny, Nx});
    py::buffer_info image_info = image.request();
    auto Image = (double *) image_info.ptr;

    backprojection(
            Sinogram,      /* Output sinogram of size Nc x Nangles */
            Nc,            /* Number of captors */
            Nangles,       /* Number of projection angles */
            s,             /* Sampling step of the captors */
            nS,            /* Interpolation degree on the Sinogram */
            t0,            /* projection of rotation center*/
            Theta,         /* Projection angles in radian */
            h,             /* Sampling step on the image */
            nI,            /* Interpolation degree on the Image */
            x0,            /* Rotation center */
            y0,
            Kernel,        /* Kernel table of size Nt x Nangles */
            Nt,            /* Number of samples in the kernel table*/
            a,             /* Maximal argument of the kernel table (0 to a) */
            Image,         /* Output image */
            Nx,            /* Size of image */
            Ny
    );

    return image;
}

PYBIND11_MODULE(csplineradon, m) {
    m.doc() = "Radon transform (and inverse) using spline convolutions discretization";

    m.def("radon", &radon, "Perform radon transform of an image");
    m.def("iradon", &iradon, "Perform inverse radon transform of a sinogram");
}
