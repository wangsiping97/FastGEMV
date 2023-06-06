#ifndef FAST_GEMV_CUH_
#define FAST_GEMV_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "utility.cuh"

#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 64
#define MAX_THREADS_PER_BLOCK 1024

///////////////////////////// GEMV //////////////////////////////
__global__ void gemv_fp16(half* mat, half* vec, half* res, unsigned int n,
                          unsigned int num_per_thread);

__global__ void gemv_quantized_int8(int8_t* mat, half* vec, half* res,
                                    unsigned int n, half scale, half zero_point,
                                    unsigned int num_per_thread);

__global__ void gemv_quantized_int4(uint4_2* mat, half* vec, half* res,
                                    unsigned int n, half scale, half zero_point,
                                    unsigned int num_per_thread);

///////////////////////////// REDUCE SUM //////////////////////////////
__device__ __forceinline__ float warpReduceSum(float sum,
                                               unsigned int threadNum);

#endif  // FAST_GEMV_CUH_