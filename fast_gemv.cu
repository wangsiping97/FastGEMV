#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "fast_gemv.cuh"

#define WARP_SIZE 32

// one block per row (gridDim.x = 1)
// thread_per_block <= WARP_SIZE
__global__ void gemv_fp16_512(half* mat, half* vec, half* res, unsigned int n,
                              unsigned int thread_per_block,
                              unsigned int num_per_thread) {
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y;
  unsigned int start_idx = threadIdx.x;
  half2* mat2 = reinterpret_cast<half2*>(mat);
  half2* vec2 = reinterpret_cast<half2*>(vec);
  half2 sum2 = make_half2(0, 0);
#pragma unroll
  for (int iter = 0; iter < num_per_thread / 2;
       iter++) {  // Assume num_per_thread is even.
    unsigned int j = start_idx + iter * thread_per_block;
    if (j < n / 2) {  // Assume n is even.
      half2 vec_val = vec2[j];
      half2 mat_val = mat2[row * (n / 2) + j];
      sum2.x += vec_val.x * mat_val.x;
      sum2.y += vec_val.y * mat_val.y;
    }
  }
  half sum = sum2.x + sum2.y;

  sum = warpReduceSum(sum, thread_per_block);

  if (tid == 0) {
    res[row] = sum;
  }
}

// thread_per_block * num_per_thread = num_per_block = n / blockDim.x
__global__ void gemv_fp16(half* mat, half* vec, half* mid_res, unsigned int n,
                          unsigned int thread_per_block,
                          unsigned int num_per_thread) {
  half sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y;
  unsigned int start_idx =
      blockIdx.x * (thread_per_block * num_per_thread) + threadIdx.x;
#pragma unroll
  for (int iter = 0; iter < num_per_thread; iter++) {
    unsigned int j = start_idx + iter * thread_per_block;
    if (j < n) {
      sum += vec[j] * mat[row * n + j];
    }
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ half warpLevelSums[WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  sum = warpReduceSum(sum, thread_per_block);
  if (laneId == 0) warpLevelSums[warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId]
                                               : (half)0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, thread_per_block / WARP_SIZE);
  if (tid == 0) {
    mid_res[row * gridDim.x + blockIdx.x] = sum;
  }
}

// block_num <= WARP_SIZE
__global__ void gemv_reduce_fp16(half* mid_res, half* res,
                                 unsigned int block_num) {
  half sum = 0;
  // each thread loads one element from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y;
  if (tid < block_num) {
    sum = mid_res[row * blockDim.x + tid];
  }
  sum = warpReduceSum(sum, block_num);
  if (tid == 0) {
    res[row] = sum;
  }
}

///////////////////////////// UTILITIES //////////////////////////////

__device__ __forceinline__ half warpReduceSum(half sum,
                                              unsigned int blockSize) {
  if (blockSize >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (blockSize >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (blockSize >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (blockSize >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (blockSize >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}

__global__ void generate_random_numbers(half* numbers, int Np) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < Np) {
    curandState state;
    curand_init(clock64(), i, 0, &state);
    numbers[i] = __float2half(curand_uniform(&state));
  }
}

__global__ void generate_numbers(half* numbers, int Np) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < Np) {
    numbers[i] = __float2half(i / 100.0);
  }
}

__global__ void check_correctness(half* mat, half* vec, half* res, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float result = 0;
    for (int j = 0; j < n; ++j) {
      result += __half2float(mat[idx * n + j]) * __half2float(vec[j]);
    }
    half half_result = __float2half(result);
    if (res[idx] != half_result) {
      float diff = __half2float(res[idx]) - __half2float(half_result);
      printf("!!![idx=%d] %f != %f, diff=%f\n", idx, __half2float(res[idx]),
             __half2float(result), diff);
    }
  }
}

// one thread for one dot product
__global__ void gemv_naive(half* mat, half* vec, half* res, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float result = 0;
    for (int j = 0; j < n; ++j) {
      result += __half2float(mat[idx * n + j]) * __half2float(vec[j]);
    }
    res[idx] = __float2half(result);
  }
}