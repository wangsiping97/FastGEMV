#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "fast_gemv.cuh"

#define WARP_SIZE 32

// thread_per_block = blockDim.x
// blockDim.y <= 32
__global__ void gemv_fp16_single_stage(half* mat, half* vec, half* res, unsigned int n,
                              unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  float4* mat4 = reinterpret_cast<float4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      float4 mat_val = mat4[row * (n >> 3) + j];
      const half2* vec_h1 = (half2*)&vec_val.x;
      const half2* vec_h2 = (half2*)&vec_val.y;
      const half2* vec_h3 = (half2*)&vec_val.z;
      const half2* vec_h4 = (half2*)&vec_val.w;
      const half2* mat_h1 = (half2*)&mat_val.x;
      const half2* mat_h2 = (half2*)&mat_val.y;
      const half2* mat_h3 = (half2*)&mat_val.z;
      const half2* mat_h4 = (half2*)&mat_val.w;
      sum += static_cast<float>(vec_h1->x) * static_cast<float>(mat_h1->x);
      sum += static_cast<float>(vec_h1->y) * static_cast<float>(mat_h1->y);
      sum += static_cast<float>(vec_h2->x) * static_cast<float>(mat_h2->x);
      sum += static_cast<float>(vec_h2->y) * static_cast<float>(mat_h2->y);
      sum += static_cast<float>(vec_h3->x) * static_cast<float>(mat_h3->x);
      sum += static_cast<float>(vec_h3->y) * static_cast<float>(mat_h3->y);
      sum += static_cast<float>(vec_h4->x) * static_cast<float>(mat_h4->x);
      sum += static_cast<float>(vec_h4->y) * static_cast<float>(mat_h4->y);
    }
  }

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      res[row] = __float2half(sum);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[32][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[threadIdx.y][laneId] : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}

// thread_per_block = blockDim.x
// blockDim.y <= 32
__global__ void gemv_fp16_multi_stage(half* mat, half* vec, half* mid_res,
                                unsigned int n, unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx =
      blockIdx.x * (blockDim.x * num_per_thread) / 8 + threadIdx.x;
  float4* mat4 = reinterpret_cast<float4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      float4 mat_val = mat4[row * (n >> 3) + j];
      const half2* vec_h1 = (half2*)&vec_val.x;
      const half2* vec_h2 = (half2*)&vec_val.y;
      const half2* vec_h3 = (half2*)&vec_val.z;
      const half2* vec_h4 = (half2*)&vec_val.w;
      const half2* mat_h1 = (half2*)&mat_val.x;
      const half2* mat_h2 = (half2*)&mat_val.y;
      const half2* mat_h3 = (half2*)&mat_val.z;
      const half2* mat_h4 = (half2*)&mat_val.w;
      sum += static_cast<float>(vec_h1->x) * static_cast<float>(mat_h1->x);
      sum += static_cast<float>(vec_h1->y) * static_cast<float>(mat_h1->y);
      sum += static_cast<float>(vec_h2->x) * static_cast<float>(mat_h2->x);
      sum += static_cast<float>(vec_h2->y) * static_cast<float>(mat_h2->y);
      sum += static_cast<float>(vec_h3->x) * static_cast<float>(mat_h3->x);
      sum += static_cast<float>(vec_h3->y) * static_cast<float>(mat_h3->y);
      sum += static_cast<float>(vec_h4->x) * static_cast<float>(mat_h4->x);
      sum += static_cast<float>(vec_h4->y) * static_cast<float>(mat_h4->y);
    }
  }

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      mid_res[row * gridDim.x + blockIdx.x] = __float2half(sum);
    }
    return;
  }

  // Shared mem for partial sums (one per warp in the block)
  static __shared__ float warpLevelSums[32][WARP_SIZE];
  const int laneId = threadIdx.x % WARP_SIZE;
  const int warpId = threadIdx.x / WARP_SIZE;
  if (laneId == 0) warpLevelSums[threadIdx.y][warpId] = sum;
  __syncthreads();
  // read from shared memory only if that warp existed
  sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[threadIdx.y][laneId] : 0.0;
  // Final reduce using first warp
  if (warpId == 0) sum = warpReduceSum(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    mid_res[row * gridDim.x + blockIdx.x] = __float2half(sum);
  }
}

// block_num <= WARP_SIZE
__global__ void gemv_reduce_fp16(half* mid_res, half* res,
                                 unsigned int block_num) {
  float sum = 0;
  // each thread loads one element from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid < block_num) {
    sum = mid_res[row * blockDim.x + tid];
  }
  sum = warpReduceSum(sum, block_num);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}

///////////////////////////// UTILITIES //////////////////////////////

__device__ __forceinline__ float warpReduceSum(float sum,
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
    numbers[i] = __float2half(i / 1000.0);
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
    float diff = __half2float(res[idx]) - __half2float(half_result);
    float delta = 0.125 * n / 512;
    if (diff > delta || diff < -delta) {
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