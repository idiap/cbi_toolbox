//
// Created by fmarelli on 18/07/19.
//

#include "cudaradon.h"


bool is_cuda_available() {
        auto dev_list = compatible_cuda_devices();
        return !dev_list.empty();
}

PYBIND11_MODULE(cudaradon, m) {

    m.doc() = "Radon transform (and inverse) using spline convolutions discretization with CUDA acceleration.";

    m.def("radon_cuda", &radon_cuda, "Perform radon transform of an image with GPU acceleration.");
    m.def("iradon_cuda", &iradon_cuda, "Perform inverse radon transform (back-projection) of a sinogram with GPU acceleration.");
    m.def("is_cuda_available", &is_cuda_available, "Check if a compatible CUDA device is available, raise runtime_error if not.");

}
