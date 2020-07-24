//
// Created by Francois Marelli on 16/04/19.
//


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstring>

namespace py = pybind11;

#define MAX(A, B)  ((A) > (B) ? (A) : (B))
#define MIN(A, B)  ((A) < (B) ? (A) : (B))

void zero(
        py::buffer_info &array_info
) {
    std::memset(array_info.ptr, 0, array_info.size * sizeof(double));
}



void radon_inner(
        double *image_ptr,           /* Image (shape {Nz, Nx, Ny}), y being the rotation axis and z the depth*/
        const long Nz,
        const long Nx,
        const long Ny,
        const double z0,             /* Rotation center Z in image coordinates */
        const double x0,             /* Rotation center X in image coordinates */
        const double h,              /* Sampling step on the image (pixel size) */
        const long nI,               /* Interpolation degree on the Image */
        double *sinogram_ptr,        /* Sinogram (shape (NAngles, Nc, Ny)*/
        const long NAngles,
        const long Nc,
        const double t0,             /* Projection of rotation center */
        const double s,              /* Sampling step of the captors (sinogram "pixel size") */
        const long nS,               /* Interpolation degree on the sinogram */
        const double *kernel_ptr,          /* Kernel table (shape {NAngles, Nt}) */
        const long Nt,
        const double tabFact,        /* Sampling step of the kernel */
        double *theta_ptr,           /* Projection angles in radian */
        bool backprojection          /* Perform a back-projection */
) {

#pragma omp parallel for
    // iterate over the projection angles
    for (long i_angle = 0; i_angle < NAngles; i_angle++) {
        double co = cos(theta_ptr[i_angle]);
        double si = sin(theta_ptr[i_angle]);

        // compute the half-width of the spline kernels with respect to the angle
        double aTheta = (double) (nI + 1L) / 2.0 * (fabs(si) + fabs(co)) * h + (double) (nS + 1L) / 2.0 * s;

        // iterate over the width of the image
        for (long i_x = 0; i_x < Nx; i_x++) {
            // compute the x coordinate of the point on the image using the pixel size
            // and its projection on the sinogram using only its x coordinate
            double x = (double)i_x * h;
            double tTemp = (x - x0) * co + t0;

            // iterate over the height of the image
            for (long i_z = 0; i_z < Nz; i_z++) {
                // compute the z coordinate of the point on the image using the pixel size
                // and its projection on the sinogram
                double z = (double)i_z * h;
                double t = tTemp + (z - z0) * si;

                // compute the range of sinogram elements impacted by this point and its spline kernel
                long iMin = MAX(0L, (long) (ceil((t - aTheta) / s)));
                long iMax = MIN(Nc - 1L, (long) (floor((t + aTheta) / s)));

                // iterate over the affected sinogram values
                for (long i_sinogram = iMin; i_sinogram <= iMax; i_sinogram++) {
                    // compute the position of the point in its spline kernel
                    double xi = fabs((double) i_sinogram * s - t);
                    long idx = (long) (floor(xi * tabFact + 0.5));

                    for (long i_y = 0; i_y < Ny; i_y++){

                        auto image_index = i_z * Nx * Ny + i_x * Ny + i_y;
                        auto kernel_index = i_angle * Nt + idx;
                        auto sinogram_index = i_angle * Nc * Ny + i_sinogram * Ny + i_y;

                        if(backprojection){
#pragma omp atomic update
                            image_ptr[image_index] += kernel_ptr[kernel_index] * sinogram_ptr[sinogram_index];
                        }
                        else{
                            // update the sinogram
                            sinogram_ptr[sinogram_index] += kernel_ptr[kernel_index] * image_ptr[image_index];
                        }
                    }
                }
            }
        }
    }
}

py::array_t<double, py::array::c_style> radon(
        py::array_t<double, py::array::c_style> &image,
        const double h,
        const long nI,
        const double z0,
        const double x0,
        py::array_t<double, py::array::c_style> &theta,
        py::array_t<double, py::array::c_style> &kernel,
        double a,
        const long Nc,
        const double s,
        const long nS,
        const double t0
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

    const long NAngles = theta_info.shape[0];

    if (NAngles != kernel_info.shape[0]) {
        throw py::value_error("The kernel must have NAngles rows.");
    }

    py::array::ShapeContainer shape = {NAngles, Nc, image_info.shape[2]};
    auto sinogram = py::array_t<double, py::array::c_style>(shape);
    auto sinogram_info = sinogram.request();

    auto *theta_ptr = reinterpret_cast<double *>(theta_info.ptr);
    auto *kernel_ptr = reinterpret_cast<double *>(kernel_info.ptr);
    auto *image_ptr = reinterpret_cast<double *>(image_info.ptr);
    auto *sinogram_ptr = reinterpret_cast<double *>(sinogram_info.ptr);

    // initialize the sinogram
    zero(sinogram_info);

    const long Nt = kernel_info.shape[1];

    const long Nz = image_info.shape[0];
    const long Nx = image_info.shape[1];
    const long Ny = image_info.shape[2];

    const double tabFact = (double) (Nt - 1L) / a;

    radon_inner(
            image_ptr, Nz, Nx, Ny, z0, x0, h, nI,
            sinogram_ptr, NAngles, Nc, t0, s, nS,
            kernel_ptr, Nt, tabFact, theta_ptr,
            false
    );

    return sinogram;
}

py::array_t<double, py::array::c_style> iradon(
        py::array_t<double, py::array::c_style> &sinogram,
        const double s,
        const long nS,
        const double t0,
        py::array_t<double, py::array::c_style> &theta,
        py::array_t<double, py::array::c_style> &kernel,
        const double a,
        const long Nz,
        const long Nx,
        const double h,
        const long nI,
        const double z0,
        const double x0
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

    const long NAngles = sinogram_info.shape[0];

    if (NAngles != theta_info.size) {
        throw py::value_error("The number of angles in theta in incompatible with the sinogram.");
    }

    if (nI < -1L) {
        throw py::value_error("nI must be greater or equal to -1.");
    }

    if (NAngles != kernel_info.shape[0]) {
        throw py::value_error("The kernel table must have NAngles rows.");
    }

    if (a < 0) {
        throw py::value_error("a, the max argument of the lookup table must be positive.");
    }

    if (Nx < 1L) {
        throw py::value_error("Nx must at least be 1.");
    }
    if (Nz < 1L) {
        throw py::value_error("Nz must at least be 1.");
    }


    const long Ny = sinogram_info.shape[2];
    py::array::ShapeContainer shape = {Nz, Nx, Ny};
    auto image = py::array_t<double, py::array::c_style>(shape);
    auto image_info = image.request();


    auto *theta_ptr = reinterpret_cast<double *>(theta_info.ptr);
    auto *sinogram_ptr = reinterpret_cast<double *>(sinogram_info.ptr);
    auto *kernel_ptr = reinterpret_cast<double *>(kernel_info.ptr);
    auto *image_ptr = reinterpret_cast<double *>(image_info.ptr);

    const long Nc = sinogram_info.shape[1];
    const long Nt = kernel_info.shape[1];

    const double tabFact = (double) (Nt - 1L) / a;

    // initialize the image
    zero(image_info);

    radon_inner(
            image_ptr, Nz, Nx, Ny, z0, x0, h, nI,
            sinogram_ptr, NAngles, Nc, t0, s, nS,
            kernel_ptr, Nt, tabFact, theta_ptr,
            true
    );

    return image;
}


PYBIND11_MODULE(ompradon, m) {

    m.doc() = "Radon transform (and inverse) using spline convolutions discretization and OMP acceleration.";

    m.def("radon", &radon, "Perform radon transform of an image with OMP acceleration.");
    m.def("iradon", &iradon, "Perform inverse radon transform (back-projection) of a sinogram with OMP acceleration.");
}
