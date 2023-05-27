#include <stdio.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <cassert>
#include <chrono>

#include "simple_tensor.h"

///////////////////////////// KERNELS //////////////////////////////

__global__ void generate_random_numbers(half* numbers, int Np) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < Np) {
        curandState state;
        curand_init(clock64(), i, 0, &state);
        numbers[i] = __float2half(curand_uniform(&state));
    }
}

__global__ void check_correctness(half* mat, half* vec, half* res, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        half result = 0;
        for (int j = 0; j < n; ++j) {
            result += mat[idx * n + j] * vec[j];
        }
        if (res[idx] != result) {
            float diff = __half2float(res[idx]) -  __half2float(result);
            printf("!!![idx=%d] %f != %f, diff=%f\n", idx, __half2float(res[idx]), __half2float(result), diff);
        }
    }
}

// one thread for one dot product
__global__ void gemv_naive(half* mat, half* vec, half* res, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        half result = 0;
        for (int j = 0; j < n; ++j) {
            result += mat[idx * n + j] * vec[j];
        }
        res[idx] = result;
    }
}

///////////////////////////// SOLVER //////////////////////////////

SimpleTensor SimpleTensor::solve_gemv(const SimpleTensor& other) const {
    assert(width_ == other.height_);
    assert(other.width_ == 1);
    SimpleTensor result(height_, 1);
    // start timing
    cudaEvent_t startEvent, stopEvent; 
    cudaEventCreate(&startEvent); 
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);
    // launch naive kernel
    // TODO: optimize
    int threads_per_block = 256;
    int num_blocks = (width_ + threads_per_block - 1) / threads_per_block;
    gemv_naive<<<num_blocks, threads_per_block>>>(data_, other.data_, result.data_, width_);
    // Record the stop event
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("Kernel time: %.3f µs\n", elapsedTime * 1000);
    return result;
}

void SimpleTensor::reset() {
    unsigned int total_elements = height_ * width_;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    generate_random_numbers<<<num_blocks, threads_per_block>>>(data_, total_elements);
    checkCudaErrors(cudaPeekAtLastError());
}

void SimpleTensor::to_host(half* host_data, unsigned n) {
    unsigned int total_elements = height_ * width_;
    assert(n <= total_elements);
    checkCudaErrors(cudaMemcpy(host_data, data_, n * sizeof(half), cudaMemcpyDeviceToHost));
}

void test_gemv(unsigned int size) {
    cudaSetDevice(0);
    // generate data
    SimpleTensor mat = SimpleTensor(size, size);
    SimpleTensor vec = SimpleTensor(size, 1);
    mat.reset();
    vec.reset();

    // compute the dot product
    printf("solving...\n");
    SimpleTensor res = mat.solve_gemv(vec);

    // check correctness
    printf("checking...\n");
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    check_correctness<<<num_blocks, threads_per_block>>>(mat.device_data(), vec.device_data(), res.device_data(), size);
    printf("checked\n");
}
