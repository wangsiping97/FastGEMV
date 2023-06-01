#ifndef FAST_GEMV_CUH_
#define FAST_GEMV_CUH_

#include "simple_tensor.h"

__global__ void generate_numbers(half* numbers, int Np);
__global__ void generate_random_numbers(half* numbers, int Np);

__global__ void check_correctness(half* mat, half* vec, half* res, int n);

// one thread for one dot product
__global__ void gemv_naive(half* mat, half* vec, half* res, int n);

__global__ void gemv_fp16_single_stage(half* mat, half* vec, half* res, unsigned int n,
                              unsigned int num_per_thread);

__global__ void gemv_fp16_multi_stage(half* mat, half* vec, half* mid_res,
                                unsigned int n, unsigned int num_per_thread);

__global__ void gemv_reduce_fp16(half* mid_res, half* res,
                                 unsigned int block_num);

__device__ __forceinline__ float warpReduceSum(float sum, unsigned int blockSize);

#endif  // FAST_GEMV_CUH_