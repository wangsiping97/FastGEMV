#ifndef FAST_GEMV_QUANTIZED_CUH_
#define FAST_GEMV_QUANTIZED_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void gemv_quantized_int8_single_stage(int8_t* mat, half* vec, half* res, unsigned int n, half scale, half zero_point,
                              unsigned int num_per_thread);

__global__ void generate_random_int8_numbers(int8_t* numbers, int Np);

__global__ void check_quantized_correctness(int8_t* mat, half* vec, half* res, half scale, half zero_point, int n);

__device__ __forceinline__ float warpReduceSum2(float sum, unsigned int blockSize);

#endif  // FAST_GEMV_QUANTIZED_CUH_