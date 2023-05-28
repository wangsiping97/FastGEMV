#ifndef FAST_GEMV_CUH_
#define FAST_GEMV_CUH_

#include "simple_tensor.h"

__global__ void generate_random_numbers(half* numbers, int Np);

__global__ void check_correctness(half* mat, half* vec, half* res, int n);

// one thread for one dot product
__global__ void gemv_naive(half* mat, half* vec, half* res, int n);

__global__ void gemv_fp16_128(half* mat, half* vec, half* res, int n);

#endif  // FAST_GEMV_CUH_