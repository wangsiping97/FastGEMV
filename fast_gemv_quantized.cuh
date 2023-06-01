#ifndef FAST_GEMV_QUANTIZED_CUH_
#define FAST_GEMV_QUANTIZED_CUH_

#include "simple_tensor.h"

__global__ void init_table_int8(half* vec, float* table, unsigned int n, float scale, int16_t zero_point);

__global__ void gemv_quantized_int8_single_stage(int8_t* mat, half* res, float* table, unsigned int n,
                                                  unsigned int num_per_thread);

__global__ void generate_random_int8_numbers(int8_t* numbers, int Np);

__global__ void check_quantized_correctness(int8_t* mat, half* vec, half* res, float scale, int16_t zero_point, int n);

__device__ __forceinline__ float warpReduceSum2(float sum,
                                               unsigned int blockSize);

#endif  // FAST_GEMV_QUANTIZED_CUH_