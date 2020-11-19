//
// Created by fmarelli on 05/07/19.
//
#include "cradon.h"

#include <cuda_runtime.h>


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
        throw std::runtime_error("No CUDA device with compute capability >=6.0 found.");
    }

    return dev_list;
}


double *numpy_to_cuda(py::buffer_info &array_info) {
    double *cuda_ptr;
    double *cpu_ptr = reinterpret_cast<double *>(array_info.ptr);

    long bytes = array_info.size * sizeof(double);

    checkCuda(cudaMalloc(&cuda_ptr, bytes));
    checkCuda(cudaMemcpy(cuda_ptr, cpu_ptr, bytes, cudaMemcpyHostToDevice));
    return cuda_ptr;
}


py::array_t<double, py::array::c_style> cuda_to_numpy(py::array::ShapeContainer &shape, double *cuda_ptr) {
    auto numpy_array = py::array_t<double, py::array::c_style>(shape);
    auto array_info = numpy_array.request();

    double *cpu_ptr = reinterpret_cast<double *>(array_info.ptr);

    long bytes = array_info.size * sizeof(double);
    checkCuda(cudaMemcpy(cpu_ptr, cuda_ptr, bytes, cudaMemcpyDeviceToHost));

    return numpy_array;
}

int optimal_threads(long max_threads, long job_size) {
    return MIN(max_threads, job_size);
}

/*
 * Precompute cosine, sine and atheta for each angle.
 * Stored in trigo (cosine, sine, atheta).
 * Grid-stride loop: blocks and threads for angles.
 */
__global__
void precompute_trigo(
        const double h,            /* Sampling step on the image (pixel size) */
        const long nI,             /* Interpolation degree on the Image */
        const long NAngles,        /* Number of angles in the sinogram (shape[0]) */
        const double s,            /* Sampling step of the captors (sinogram "pixel size") */
        const long nS,             /* Interpolation degree on the sinogram */
        double *theta,             /* Projection angles in radian */
        double *trigo              /* Array containing cosine, sine and atheta for each angle shape {NAngles, 3}*/
) {

    // iterate over the projection angles using blocks and threads alike
    for (long i_angle = blockIdx.x * blockDim.x + threadIdx.x; i_angle < NAngles; i_angle += blockDim.x * gridDim.x) {

        double co = cos(theta[i_angle]);
        double si = sin(theta[i_angle]);

        double aTheta = (double) (nI + 1L) / 2.0 * (fabs(si) + fabs(co)) * h + (double) (nS + 1L) / 2.0 * s;

        long index = i_angle * 3;
        trigo[index] = co;
        trigo[index + 1] = si;
        trigo[index + 2] = aTheta;
    }
}


/*
 * Precompute the minimum and maximum indexes of sinogram values impacted by each pixel in the image, and the projected coordinates.
 * Stored in sino_bounds (min, max) and t_coord.
 * Grid-stride loop: blocks for angles (NAngles) , threads for image pixels (Nx * Ny).
 */
__global__
void precompute_radon(
        const long Nz,             /* Image Z size (shape[0]) */
        const long Nx,             /* Image X size (shape[1]) */
        const double h,            /* Sampling step on the image (pixel size) */
        const double z0,           /* Rotation center Z in image coordinates */
        const double x0,           /* Rotation center X in image coordinates */
        const long NAngles,        /* Number of angles in the sinogram (shape[0]) */
        const long Nc,             /* Number of captors in the sinogram (shape[1]) */
        const double s,            /* Sampling step of the captors (sinogram "pixel size") */
        const double t0,           /* Projection of rotation center */
        double *trigo,             /* Array containing cosine, sine and atheta for each angle (shape {NAngles, 3}) */
        long *sino_bounds,         /* Indexes of sinogram impact for all pixels in the image (shape {A, X, Z, 2}) */
        double *t_coords           /* Projected coordinates on the sinogram (shape {A, X, Z}) */
) {

    // iterate over the projection angles using blocks
    for (long i_angle = blockIdx.x; i_angle < NAngles; i_angle += gridDim.x) {

        long index_angle = i_angle * 3;

        double co = trigo[index_angle];
        double si = trigo[index_angle + 1];
        double atheta = trigo[index_angle + 2];

        // iterate over the image using threads
        for (long id = threadIdx.x; id < Nx * Nz; id += blockDim.x) {
            long i_x = id / Nz;
            long i_z = id % Nz;

            double x = i_x * h;
            double z = i_z * h;

            // compute the projected coordinate on the sinogram
            double t = t0 + ((x - x0) * co) + ((z - z0) * si);

            // compute the range of sinogram elements impacted by this point and its spline kernel
            long imin = MAX(0L, (long) (ceil((t - atheta) / s)));
            long imax = MIN(Nc - 1L, (long) (floor((t + atheta) / s)));

            // store in the relevant matrices
            auto t_index = i_angle * Nx * Nz + i_x * Nz + i_z;
            t_coords[t_index] = t;

            auto bound_index = t_index * 2;
            sino_bounds[bound_index] = imin;
            sino_bounds[bound_index + 1] = imax;
        }
    }
}

/*
 * Compute the (inverse) radon transform.
 * Grid-stide loop: blocks for angles (NAngles), threads for depth (Nz)
 */
__global__
void cuda_radontransform(
        double *image,               /* Image (shape {Nz, Nx, Ny})*/
        const long Nz,
        const long Nx,
        const long Ny,
        double *sinogram,            /* Sinogram (shape (NAngles, Nc, Ny)*/
        const long NAngles,
        const long Nc,
        const double s,              /* Sampling step of the captors (sinogram "pixel size") */
        double *kernel,              /* Kernel table (shape {NAngles, Nt}) */
        const long Nt,
        const double tabfact,        /* Sampling step of the kernel */
        long *sino_bounds,           /* Indexes of sinogram impact for all pixels in the image (shape {A, X, Z, 2}) */
        double *t_coords,            /* Projected coordinates on the sinogram (shape {A, X, Z}) */
        bool backprojection          /* Perform a back-projection */
) {

    // iterate over the projection angles
    for (long i_angle = blockIdx.x; i_angle < NAngles; i_angle += gridDim.x) {

        // iterate over the width of the image
        for (long i_x = 0; i_x < Nx; i_x++) {

            // iterate over the height of the image
            for (long i_z = 0; i_z < Nz; i_z++) {

                // fetch the projected coordinate
                auto t_index = i_angle * Nx * Nz + i_x * Nz + i_z;
                auto t = t_coords[t_index];

                // fetch the sinogram bounds
                auto bounds_index = t_index * 2;
                auto imin = sino_bounds[bounds_index];
                auto imax = sino_bounds[bounds_index + 1];

                // iterate over the affected sinogram values
                for (long i_sino = imin; i_sino <= imax; i_sino++) {
                    // compute the position of the point in its spline kernel
                    double xi = fabs((double) i_sino * s - t);
                    long idx = (long) (floor(xi * tabfact + 0.5));

                    for (long i_y = threadIdx.x; i_y < Ny; i_y += blockDim.x) {

                        auto image_index = i_z * Nx * Ny + i_x * Ny + i_y;
                        auto kernel_index = i_angle * Nt + idx;
                        auto sinogram_index = i_angle * Nc * Ny + i_sino * Ny + i_y;

                        if (backprojection) {
                            // update the image
                            atomicAdd(image + image_index, kernel[kernel_index] * sinogram[sinogram_index]);
                        } else {
                            // update the sinogram
                            sinogram[sinogram_index] += (kernel[kernel_index] * image[image_index]);
                        }
                    }
                }
            }
        }
    }
}


py::array_t<double, py::array::c_style> radon_cuda(
        py::array_t<double, py::array::c_style> &image,
        const double h,
        const long nI,
        const double z0,
        const double x0,
        py::array_t<double, py::array::c_style> &theta,
        py::array_t<double, py::array::c_style> &kernel,
        const double a,
        const long Nc,
        const double s,
        const long nS,
        const double t0
) {

    auto image_info = image.request();
    auto kernel_info = kernel.request();
    auto theta_info = theta.request();
    
    const long NAngles = theta_info.shape[0];

    auto dev_list = compatible_cuda_devices();
    auto device_id = dev_list[0];
    checkCuda(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    auto max_threads = prop.maxThreadsPerBlock;

    double *cuda_theta = numpy_to_cuda(theta_info);

    double *cuda_trigo;
    checkCuda(cudaMalloc(&cuda_trigo, NAngles * 3 * sizeof(double)));

    auto n_threads = optimal_threads(max_threads, NAngles);

    precompute_trigo << < 1, n_threads >> > (
            h,
                    nI,
                    NAngles,
                    s,
                    nS,
                    cuda_theta,
                    cuda_trigo
    );

    const long Nz = image_info.shape[0];
    const long Nx = image_info.shape[1];
    const long Ny = image_info.shape[2];

    long *cuda_sino_bounds;
    double *cuda_t_coords;

    n_threads = optimal_threads(max_threads, Nx * Nz);

    checkCuda(cudaFree(cuda_theta));

    checkCuda(cudaMalloc(&cuda_sino_bounds, NAngles * Nx * Nz * 2 * sizeof(long)));
    checkCuda(cudaMalloc(&cuda_t_coords, NAngles * Nx * Nz * sizeof(double)));

    precompute_radon << < NAngles, n_threads >> > (
                    Nz,
                    Nx,
                    h,
                    z0,
                    x0,
                    NAngles,
                    Nc,
                    s,
                    t0,
                    cuda_trigo,
                    cuda_sino_bounds,
                    cuda_t_coords
    );

    n_threads = optimal_threads(max_threads, Nz);

    double *cuda_sinogram;
    long sinogram_bytes = NAngles * Nc * Ny * sizeof(double);
    auto Nt = kernel_info.shape[1];
    double tabfact = (double) (Nt - 1L) / a;

    checkCuda(cudaFree(cuda_trigo));

    // TODO use streams to accelerate data loading
    // TODO combine implementations with radon and iradon (very similar!)
    double *cuda_image = numpy_to_cuda(image_info);
    double *cuda_kernel = numpy_to_cuda(kernel_info);

    checkCuda(cudaMalloc(&cuda_sinogram, sinogram_bytes));
    checkCuda(cudaMemset(cuda_sinogram, 0, sinogram_bytes));

    cuda_radontransform <<<NAngles, n_threads>>>(
            cuda_image,
                    Nz,
                    Nx,
                    Ny,
                    cuda_sinogram,
                    NAngles,
                    Nc,
                    s,
                    cuda_kernel,
                    kernel_info.shape[1],
                    tabfact,
                    cuda_sino_bounds,
                    cuda_t_coords,
                    false
    );

    checkCuda(cudaFree(cuda_image));

    checkCuda(cudaFree(cuda_kernel));
    checkCuda(cudaFree(cuda_t_coords));
    checkCuda(cudaFree(cuda_sino_bounds));

    py::array::ShapeContainer shape = {NAngles, Nc, Ny};

    auto sinogram = cuda_to_numpy(shape, cuda_sinogram);
    checkCuda(cudaFree(cuda_sinogram));

    return sinogram;
}

py::array_t<double, py::array::c_style> iradon_cuda(
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

    const long NAngles = sinogram_info.shape[0];


    auto dev_list = compatible_cuda_devices();
    auto device_id = dev_list[0];
    checkCuda(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    auto max_threads = prop.maxThreadsPerBlock;

    double *cuda_theta = numpy_to_cuda(theta_info);

    double *cuda_trigo;
    checkCuda(cudaMalloc(&cuda_trigo, NAngles * 3 * sizeof(double)));

    auto n_threads = optimal_threads(max_threads, NAngles);

    precompute_trigo << < 1, n_threads >> > (
            h,
                    nI,
                    NAngles,
                    s,
                    nS,
                    cuda_theta,
                    cuda_trigo
    );


    const long Ny = sinogram_info.shape[2];
    const long  Nc = sinogram_info.shape[1];

    long *cuda_sino_bounds;
    double *cuda_t_coords;

    n_threads = optimal_threads(max_threads, Nx * Nz);

    checkCuda(cudaFree(cuda_theta));

    checkCuda(cudaMalloc(&cuda_sino_bounds, NAngles * Nx * Nz * 2 * sizeof(long)));
    checkCuda(cudaMalloc(&cuda_t_coords, NAngles * Nx * Nz * sizeof(double)));

    precompute_radon << < NAngles, n_threads >> > (
                    Nz,
                    Nx,
                    h,
                    z0,
                    x0,
                    NAngles,
                    Nc,
                    s,
                    t0,
                    cuda_trigo,
                    cuda_sino_bounds,
                    cuda_t_coords
    );

    n_threads = optimal_threads(max_threads, Ny);

    double *cuda_image;
    long image_bytes = Nz * Nx * Ny * sizeof(double);
    auto Nt = kernel_info.shape[1];
    double tabfact = (double) (Nt - 1L) / a;

    checkCuda(cudaFree(cuda_trigo));

    // TODO use streams to accelerate data loading
    double *cuda_sinogram = numpy_to_cuda(sinogram_info);
    double *cuda_kernel = numpy_to_cuda(kernel_info);

    checkCuda(cudaMalloc(&cuda_image, image_bytes));
    checkCuda(cudaMemset(cuda_image, 0, image_bytes));

    cuda_radontransform << < NAngles, n_threads >> > (
            cuda_image,
                    Nz,
                    Nx,
                    Ny,
                    cuda_sinogram,
                    NAngles,
                    Nc,
                    s,
                    cuda_kernel,
                    kernel_info.shape[1],
                    tabfact,
                    cuda_sino_bounds,
                    cuda_t_coords,
                    true
    );

    checkCuda(cudaFree(cuda_sinogram));

    checkCuda(cudaFree(cuda_kernel));
    checkCuda(cudaFree(cuda_t_coords));
    checkCuda(cudaFree(cuda_sino_bounds));

    py::array::ShapeContainer shape = {Nz, Nx, Ny};

    auto image = cuda_to_numpy(shape, cuda_image);
    checkCuda(cudaFree(cuda_image));

    return image;
}
