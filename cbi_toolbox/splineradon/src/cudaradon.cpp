//
// Created by fmarelli on 18/07/19.
//

#include "cradon.h"

#ifdef CUDA

bool is_cuda_available()
{
    auto dev_list = compatible_cuda_devices();
    return !dev_list.empty();
}

#else //CUDA

bool is_cuda_available()
{
    throw std::runtime_error("CUDA support is not installed.");
}

#endif //CUDA