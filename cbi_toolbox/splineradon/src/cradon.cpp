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

#include "cradon.h"

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
    const double t0,
    bool use_cuda)
{

    auto image_info = image.request();
    auto kernel_info = kernel.request();
    auto theta_info = theta.request();

    if (image_info.ndim != 3)
    {
        throw py::value_error("image must be a 3D array.");
    }

    if (theta_info.ndim != 1)
    {
        throw py::value_error("theta must be a 1D array.");
    }

    if (kernel_info.ndim != 2)
    {
        throw py::value_error("kernel must be a 2D array.");
    }

    if (nI < 0L)
    {
        throw py::value_error("nI must be greater or equal to 0.");
    }

    if (a < 0.0)
    {
        throw py::value_error("a, the maximal argument of the lookup table must be a positive.");
    }

    if (Nc < 1L)
    {
        throw py::value_error("The number of captor must at least be 1.");
    }

    if (nS < -1L)
    {
        throw py::value_error("nS must be greater of equal to -1.");
    }

    const long NAngles = theta_info.shape[0];

    if (NAngles != kernel_info.shape[0])
    {
        throw py::value_error("The kernel must have NAngles rows.");
    }

    if (use_cuda)
    {
#ifdef CUDA
        return radon_cuda(image, h, nI, z0, x0, theta, kernel, a, Nc, s, nS, t0);
#else  //CUDA
        throw std::runtime_error("CUDA support is not installed.");
#endif //CUDA
    }
    else
    {
        return radon_omp(image, h, nI, z0, x0, theta, kernel, a, Nc, s, nS, t0);
    }
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
    const double x0,
    bool use_cuda)
{

    auto sinogram_info = sinogram.request();
    auto kernel_info = kernel.request();
    auto theta_info = theta.request();

    if (sinogram_info.ndim != 3)
    {
        throw py::value_error("sinogram must be a 3D array.");
    }

    if (theta_info.ndim != 1)
    {
        throw py::value_error("theta must be a 1D array.");
    }

    if (kernel_info.ndim != 2)
    {
        throw py::value_error("kernel must be a 2D array.");
    }

    if (nS < 0L)
    {
        throw py::value_error("nS must be greater or equal to 0.");
    }

    const long NAngles = sinogram_info.shape[0];

    if (NAngles != theta_info.size)
    {
        throw py::value_error("The number of angles in theta in incompatible with the sinogram.");
    }

    if (nI < -1L)
    {
        throw py::value_error("nI must be greater or equal to -1.");
    }

    if (NAngles != kernel_info.shape[0])
    {
        throw py::value_error("The kernel table must have NAngles rows.");
    }

    if (a < 0.0)
    {
        throw py::value_error("a, the max argument of the lookup table must be positive.");
    }

    if (Nx < 1L)
    {
        throw py::value_error("Nx must at least be 1.");
    }
    if (Nz < 1L)
    {
        throw py::value_error("Nz must at least be 1.");
    }

    if (use_cuda)
    {
#ifdef CUDA
        return iradon_cuda(sinogram, s, nS, t0, theta, kernel, a, Nz, Nx, h, nI, z0, x0);
#else  //CUDA
        throw std::runtime_error("CUDA support is not installed.");
#endif //CUDA
    }
    else
    {
        return iradon_omp(sinogram, s, nS, t0, theta, kernel, a, Nz, Nx, h, nI, z0, x0);
    }
}

PYBIND11_MODULE(_cradon, m)
{
    m.doc() = "Radon transform (and inverse) using spline convolutions discretization and OMP/CUDA acceleration.";

    m.def("radon", &radon, "Perform radon transform of an image with OMP/CUDA acceleration.");
    m.def("iradon", &iradon, "Perform inverse radon transform (back-projection) of a sinogram with OMP/CUDA acceleration.");
    m.def("is_cuda_available", &is_cuda_available, "Check if a compatible CUDA device is available, raise runtime_error if not.");
}