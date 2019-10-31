//
// Created by fmarelli on 16/04/19.
//

#include "tomography_omp.h"

py::array_t<double, py::array::c_style> radon(
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
) {
  
    auto image_info = image.request();
    auto kernel_info = kernel.request();
    auto theta_info = theta.request();

    if (image_info.ndim != 3) {
        throw py::value_error("image must be a 3D array.");
    }

    if (theta_info.ndim != 1) {
        throw py::value_error("theta must be a 1D array.");
    }

    if (kernel_info.ndim != 2) {
        throw py::value_error("kernel must be a 2D array.");
    }

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

    const long Nangles = theta_info.shape[0];

    if (Nangles != kernel_info.shape[0]) {
        throw py::value_error("The kernel must have Nangle rows.");
    }

    py::array::ShapeContainer shape = {Nangles, Nc, image_info.shape[2]};
    auto sinogram = py::array_t<double, py::array::c_style>(shape);

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

py::array_t<double, py::array::c_style> iradon(
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
) {

    auto sinogram_info = sinogram.request();
    auto kernel_info = kernel.request();
    auto theta_info = theta.request();

    if (sinogram_info.ndim != 3) {
        throw py::value_error("sinogram must be a 3D array.");
    }

    if (theta_info.ndim != 1) {
        throw py::value_error("theta must be a 1D array.");
    }

    if (kernel_info.ndim != 2) {
        throw py::value_error("kernel must be a 2D array.");
    }

    if (nS < 0L) {
        throw py::value_error("nS must be greater or equal to 0.");
    }

    const long Nangles = sinogram_info.shape[0];

    if (Nangles != theta_info.size) {
        throw py::value_error("The number of angles in theta in incompatible with the sinogram.");
    }

    if (nI < -1L) {
        throw py::value_error("nI must be greater or equal to -1.");
    }

    if (Nangles != kernel_info.shape[0]) {
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
    
    py::array::ShapeContainer shape = {Ny, Nx, sinogram_info.shape[2]};
    auto image = py::array_t<double, py::array::c_style>(shape);

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


PYBIND11_MODULE(ompsplineradon, m) {
   
    m.doc() = "Radon transform (and inverse) using spline convolutions discretization";

    m.def("radon", &radon, "Perform radon transform of an image");
    m.def("iradon", &iradon, "Perform inverse radon transform (back-projection) of a sinogram");
}
