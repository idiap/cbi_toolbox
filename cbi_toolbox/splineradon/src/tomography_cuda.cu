//
// Created by fmarelli on 05/07/19.
//
#include "tomography_cuda.h"

#include <cuda_runtime.h>
#include <sm_60_atomic_functions.h>
#include <device_launch_parameters.h>

#include <math.h>

namespace py = pybind11;

// Throw std::runtime_error when a cuda error occurs
#define CUDA_EXCEPTIONS


#define MAX(A, B)  ((A) > (B) ? (A) : (B))
#define MIN(A, B)  ((A) < (B) ? (A) : (B))

inline
cudaError_t checkCuda(cudaError_t result) {
#if defined(CUDA_EXCEPTIONS)
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
#endif
    return result;
}

std::vector<int> compatible_cuda_devices() {
    int nDevices;
    checkCuda(cudaGetDeviceCount(&nDevices));

    std::vector<int> dev_list;
    cudaDeviceProp prop;

    for (int i = 0; i < nDevices; i++) {
        checkCuda(cudaGetDeviceProperties(&prop, 0));
        if (prop.major >= 6) {
            dev_list.push_back(i);
        }
    }
    if (dev_list.empty()) {
        throw std::runtime_error("No CUDA device with capability minimum 6.0 found");
    }

    return dev_list;
}

void numpy_to_cuda(py::buffer_info &array_info, double **cuda_ptr) {
    double *cpu_ptr = reinterpret_cast<double *>(array_info.ptr);

    size_t bytes = array_info.size * sizeof(double);

    checkCuda(cudaMalloc(cuda_ptr, bytes));
    checkCuda(cudaMemcpy(*cuda_ptr, cpu_ptr, bytes, cudaMemcpyHostToDevice));
}

void cuda_to_numpy(py::buffer_info &array_info, double *cuda_ptr) {
    double *cpu_ptr = reinterpret_cast<double *>(array_info.ptr);

    size_t bytes = array_info.size * sizeof(double);
    checkCuda(cudaMemcpy(cpu_ptr, cuda_ptr, bytes, cudaMemcpyDeviceToHost));
}


__global__
void cuda_radontransform(
        double *image,                       /* Image */
        size_t Nx,
        size_t Ny,
        size_t Nz,
        double h,                            /* Sampling step on the image (pixel size) */
        long nI,                             /* Interpolation degree on the Image */
        double x0,                           /* Rotation center in image coordinates */
        double y0,
        double *sinogram,                    /* Sinogram of size Nangles x Nc x Nlayers*/
        size_t Nangles,
        size_t Nc,
        double s,                            /* Sampling step of the captors (sinogram "pixel size") */
        long nS,                             /* Interpolation degree on the sinogram */
        double t0,                           /* Projection of rotation center */
        double *theta,      /* Projection angles in radian */
        double *kernel,     /* Kernel table of size Nangles x Nt */
        size_t Nt,
        double a,                            /* Maximal argument of the kernel table (0 to a) */
        bool backprojection                  /* Perform a back-projection */
) {

    //TODO compute this only once??
    double tmax = s * ((double) (Nc - 1L));
    double tabfact = (double) (Nt - 1L) / a;


    // iterate over the projection angles
    for (long i_angle = blockIdx.x; i_angle < Nangles; i_angle += gridDim.x) {

        //TODO compute this only once??
        double co = cos(theta[i_angle]);
        double si = sin(theta[i_angle]);


        double atheta = (double) (nI + 1L) / 2.0 * (fabs(si) + fabs(co)) * h + (double) (nS + 1L) / 2.0 * s;


        // TODO parallelize this?
        // iterate over the width of the image
        for (long i_x = 0; i_x < Nx; i_x++) {
            // compute the x coordinate of the point on the image using the pixel size
            // and its projection on the sinogram using only its x coordinate
            double x = i_x * h;
            double ttemp = (x - x0) * co + t0;

            // TODO parallelize this?
            // iterate over the height of the image
            for (long i_y = 0; i_y < Ny; i_y++) {
                // compute the y coordinate of the point on the image using the pixel size
                // and its projection on the sinogram
                double y = i_y * h;
                double t = ttemp + (y - y0) * si;

                // if the projection is in the sinogram (depends on the alignment of the centers {x0, y0} and t0
                if ((t > 0.0) && (t <= tmax)) {
                    // compute the range of sinogram elements impacted by this point and its spline kernel
                    long imin = MAX(0L, (long) (ceil((t - atheta) / s)));
                    long imax = MIN(Nc - 1L, (long) (floor((t + atheta) / s)));

                    // TODO parallelize this?
                    // iterate over the affected sinogram values
                    for (long i_sino = imin; i_sino <= imax; i_sino++) {
                        // compute the position of the point in its spline kernel
                        double xi = fabs((double) i_sino * s - t);
                        long idx = (long) (floor(xi * tabfact + 0.5));

                        for (long i_z = threadIdx.x; i_z < Nz; i_z += blockDim.x) {

                            auto image_index = i_y * Nx * Nz + i_x * Nz + i_z;
                            auto kernel_index = i_angle * Nt + idx;
                            auto sinogram_index = i_angle * Nc * Nz + i_sino * Nz + i_z;


                            if (backprojection) {
                                // update the image
                                atomicAdd(image + image_index, kernel[kernel_index] * sinogram[sinogram_index]);
                            } else {
                                // update the sinogram
                                atomicAdd(sinogram + sinogram_index, kernel[kernel_index] * image[image_index]);
                            }
                        }
                    }
                }
            }
        }
    }
}


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

    const long Nangles = theta.shape()[0];

    if (Nangles != kernel.shape()[0]) {
        throw py::value_error("The kernel must have Nangle rows.");
    }

    auto dev_list = compatible_cuda_devices();
    auto device_id = dev_list[0];
    checkCuda(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    auto max_threads = prop.maxThreadsPerBlock;
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device_id);

    double *cuda_image;
    numpy_to_cuda(image_info, &cuda_image);

    double *cuda_sinogram;
    size_t sinogram_bytes = Nangles * Nc * image_info.shape[2] * sizeof(double);
    checkCuda(cudaMalloc(&cuda_sinogram, sinogram_bytes));
    checkCuda(cudaMemset(cuda_sinogram, 0, sinogram_bytes));

    double *cuda_kernel;
    numpy_to_cuda(kernel_info, &cuda_kernel);

    double *cuda_theta;
    numpy_to_cuda(theta_info, &cuda_theta);

    // TODO change this for x and y?
    auto n_threads = max_threads;
    if (max_threads > image_info.shape[2]) {
        n_threads = (image_info.shape[2] / 32) * 32;
        if (!n_threads) {
            n_threads = image_info.shape[2];
        }
    }

    cuda_radontransform << < Nangles, n_threads >> > (
            cuda_image,
                    image_info.shape[1],
                    image_info.shape[0],
                    image_info.shape[2],
                    h,
                    nI,
                    x0,
                    y0,
                    cuda_sinogram,
                    Nangles,
                    Nc,
                    s,
                    nS,
                    t0,
                    cuda_theta,
                    cuda_kernel,
                    kernel_info.shape[1],
                    a,
                    false
    );

    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaFree(cuda_image));
    checkCuda(cudaFree(cuda_theta));
    checkCuda(cudaFree(cuda_kernel));

    auto sinogram = py::array_t<double, py::array::c_style>({Nangles, Nc, image_info.shape[2]});
    auto sinogram_info = sinogram.request();

    cuda_to_numpy(sinogram_info, cuda_sinogram);
    checkCuda(cudaFree(cuda_sinogram));

    return sinogram;
}
