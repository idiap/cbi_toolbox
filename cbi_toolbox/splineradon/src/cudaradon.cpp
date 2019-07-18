//
// Created by fmarelli on 18/07/19.
//

#include "tomography_cuda.h"


bool is_cuda_available() {
    try {
        auto dev_list = compatible_cuda_devices();
        return !dev_list.empty();
    }
    catch (const std::runtime_error &e) {
        std::cout << e.what() << std::endl;
        return false;
    }
}

PYBIND11_MODULE(cudaradon, m) {

    m.doc() = "Radon transform (and inverse) using spline convolutions discretization and CUDA acceleration";

    m.def("radon_cuda", &radon_cuda, "Perform radon transform of an image with GPU acceleration");
    m.def("is_cuda_available", &is_cuda_available, "Check if a compatible CUDA device is available");

}