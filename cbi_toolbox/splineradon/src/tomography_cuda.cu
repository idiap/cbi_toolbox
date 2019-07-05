//
// Created by fmarelli on 05/07/19.
//

#include <iostream>
#include <math.h>
#include "tomography_cuda.h"

__global__
void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

void test_cuda() {
    int N = 1 << 20;

    float *x, *y;

    cudaMallocManaged(&x, N * sizeof(float));
    cudaError_t error = cudaMallocManaged(&y, N * sizeof(float));

    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }

    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add<<<1, 1>>>(N, x, y);

    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++){
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    }

    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
}