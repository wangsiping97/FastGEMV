#ifndef FAST_GEMV_CUH_
#define FAST_GEMV_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define SHARED_MEM_MAX_ROWS 64

///////////////////////////// NORMAL //////////////////////////////
__global__ void gemv_fp16_single_stage(half* mat, half* vec, half* res, unsigned int n,
                              unsigned int num_per_thread);
__global__ void gemv_fp16_multi_stage(half* mat, half* vec, half* mid_res,
                                unsigned int n, unsigned int num_per_thread);

///////////////////////////// QUANTIZED-INT8 //////////////////////////////
__global__ void gemv_quantized_int8_single_stage(int8_t* mat, half* vec, half* res, unsigned int n, half scale, half zero_point,
                              unsigned int num_per_thread);
__global__ void gemv_quantized_int8_multi_stage(int8_t* mat, half* vec, half* mid_res,
                                unsigned int n, half scale, half zero_point, unsigned int num_per_thread);

///////////////////////////// QUANTIZED-INT4 //////////////////////////////
__global__ void gemv_quantized_int4_single_stage(uint4_2* mat, half* vec, half* res, unsigned int n, half scale, half zero_point,
                              unsigned int num_per_thread);
__global__ void gemv_quantized_int4_multi_stage(uint4_2* mat, half* vec, half* mid_res,
                                unsigned int n, half scale, half zero_point, unsigned int num_per_thread);

///////////////////////////// REDUCE SUM //////////////////////////////
__global__ void gemv_reduce_fp16(half* mid_res, half* res,
                                 unsigned int block_num);
__device__ __forceinline__ float warpReduceSum(float sum, unsigned int blockSize);

#endif  // FAST_GEMV_CUH_