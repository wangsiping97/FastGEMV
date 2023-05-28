#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "fast_gemv.cuh"

__global__ void gemv_fp16_128(half* mat, half* vec, half* res, int n) {}

///////////////////////////// UTILITIES //////////////////////////////

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
      float diff = __half2float(res[idx]) - __half2float(result);
      printf("!!![idx=%d] %f != %f, diff=%f\n", idx, __half2float(res[idx]),
             __half2float(result), diff);
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