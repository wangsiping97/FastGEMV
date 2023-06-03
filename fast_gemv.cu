#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "utility.cuh"
#include "fast_gemv.cuh"

#define WARP_SIZE 32

///////////////////////////// NORMAL //////////////////////////////
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

///////////////////////////// QUANTIZED-INT8 //////////////////////////////

__global__ void gemv_quantized_int8_single_stage(int8_t* mat, half* vec, half* res, unsigned int n, half scale, half zero_point,
                              unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  half4* mat4 = reinterpret_cast<half4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

  float zero_point_f = static_cast<float>(zero_point);
  float scale_f = static_cast<float>(scale);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      half4 mat_val = mat4[row * (n >> 3) + j];
      const half2* vec_h1 = (half2*)&vec_val.x;
      const half2* vec_h2 = (half2*)&vec_val.y;
      const half2* vec_h3 = (half2*)&vec_val.z;
      const half2* vec_h4 = (half2*)&vec_val.w;
      const int8_2* mat_h1 = (int8_2*)&mat_val.x;
      const int8_2* mat_h2 = (int8_2*)&mat_val.y;
      const int8_2* mat_h3 = (int8_2*)&mat_val.z;
      const int8_2* mat_h4 = (int8_2*)&mat_val.w;
      sum += static_cast<float>(vec_h1->x) * (static_cast<float>(mat_h1->x) - zero_point_f);
      sum += static_cast<float>(vec_h1->y) * (static_cast<float>(mat_h1->y) - zero_point_f);
      sum += static_cast<float>(vec_h2->x) * (static_cast<float>(mat_h2->x) - zero_point_f);
      sum += static_cast<float>(vec_h2->y) * (static_cast<float>(mat_h2->y) - zero_point_f);
      sum += static_cast<float>(vec_h3->x) * (static_cast<float>(mat_h3->x) - zero_point_f);
      sum += static_cast<float>(vec_h3->y) * (static_cast<float>(mat_h3->y) - zero_point_f);
      sum += static_cast<float>(vec_h4->x) * (static_cast<float>(mat_h4->x) - zero_point_f);
      sum += static_cast<float>(vec_h4->y) * (static_cast<float>(mat_h4->y) - zero_point_f);
    }
  }

  sum *= scale_f;

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
__global__ void gemv_quantized_int8_multi_stage(int8_t* mat, half* vec, half* mid_res,
                                unsigned int n, half scale, half zero_point, unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx =
      blockIdx.x * (blockDim.x * num_per_thread) / 8 + threadIdx.x;
  half4* mat4 = reinterpret_cast<half4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

  float zero_point_f = static_cast<float>(zero_point);
  float scale_f = static_cast<float>(scale);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      float4 vec_val = vec4[j];
      half4 mat_val = mat4[row * (n >> 3) + j];
      const half2* vec_h1 = (half2*)&vec_val.x;
      const half2* vec_h2 = (half2*)&vec_val.y;
      const half2* vec_h3 = (half2*)&vec_val.z;
      const half2* vec_h4 = (half2*)&vec_val.w;
      const int8_2* mat_h1 = (int8_2*)&mat_val.x;
      const int8_2* mat_h2 = (int8_2*)&mat_val.y;
      const int8_2* mat_h3 = (int8_2*)&mat_val.z;
      const int8_2* mat_h4 = (int8_2*)&mat_val.w;
      sum += static_cast<float>(vec_h1->x) * (static_cast<float>(mat_h1->x) - zero_point_f);
      sum += static_cast<float>(vec_h1->y) * (static_cast<float>(mat_h1->y) - zero_point_f);
      sum += static_cast<float>(vec_h2->x) * (static_cast<float>(mat_h2->x) - zero_point_f);
      sum += static_cast<float>(vec_h2->y) * (static_cast<float>(mat_h2->y) - zero_point_f);
      sum += static_cast<float>(vec_h3->x) * (static_cast<float>(mat_h3->x) - zero_point_f);
      sum += static_cast<float>(vec_h3->y) * (static_cast<float>(mat_h3->y) - zero_point_f);
      sum += static_cast<float>(vec_h4->x) * (static_cast<float>(mat_h4->x) - zero_point_f);
      sum += static_cast<float>(vec_h4->y) * (static_cast<float>(mat_h4->y) - zero_point_f);
    }
  }

  sum *= scale_f;

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

///////////////////////////// QUANTIZED-INT4 //////////////////////////////

// based on previous experiments, num_per_thread can >= 16
__global__ void gemv_quantized_int4_single_stage(uint4_2* mat, half* vec, half* res, unsigned int n, half scale, half zero_point,
                              unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  uint4_2_4* mat4 = reinterpret_cast<uint4_2_4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

  float zero_point_f = static_cast<float>(zero_point);
  float scale_f = static_cast<float>(scale);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 4; iter++) {
    unsigned int j = 2 * (start_idx + iter * blockDim.x);
    if (j < n >> 3) {
      float4 vec_val_1 = vec4[j];  // 8 half
      float4 vec_val_2 = vec4[j + 1];
      const half2* vec_h1 = (half2*)&vec_val_1.x;
      const half2* vec_h2 = (half2*)&vec_val_1.y;
      const half2* vec_h3 = (half2*)&vec_val_1.z;
      const half2* vec_h4 = (half2*)&vec_val_1.w;
      const half2* vec_h5 = (half2*)&vec_val_2.x;
      const half2* vec_h6 = (half2*)&vec_val_2.y;
      const half2* vec_h7 = (half2*)&vec_val_2.z;
      const half2* vec_h8 = (half2*)&vec_val_2.w;

      uint4_2_4 mat_val_1 = mat4[row * (n >> 3) + j];
      uint4_2_4 mat_val_2 = mat4[row * (n >> 3) + j + 1];
      const uint4_2* mat_h1 = (uint4_2*)&mat_val_1.x;
      const uint4_2* mat_h2 = (uint4_2*)&mat_val_1.y;
      const uint4_2* mat_h3 = (uint4_2*)&mat_val_1.z;
      const uint4_2* mat_h4 = (uint4_2*)&mat_val_1.w;
      const uint4_2* mat_h5 = (uint4_2*)&mat_val_2.x;
      const uint4_2* mat_h6 = (uint4_2*)&mat_val_2.y;
      const uint4_2* mat_h7 = (uint4_2*)&mat_val_2.z;
      const uint4_2* mat_h8 = (uint4_2*)&mat_val_2.w;

      sum += static_cast<float>(vec_h1->x) * (static_cast<float>(mat_h1->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h1->y) * (static_cast<float>(mat_h1->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h2->x) * (static_cast<float>(mat_h2->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h2->y) * (static_cast<float>(mat_h2->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h3->x) * (static_cast<float>(mat_h3->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h3->y) * (static_cast<float>(mat_h3->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h4->x) * (static_cast<float>(mat_h4->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h4->y) * (static_cast<float>(mat_h4->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h5->x) * (static_cast<float>(mat_h5->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h5->y) * (static_cast<float>(mat_h5->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h6->x) * (static_cast<float>(mat_h6->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h6->y) * (static_cast<float>(mat_h6->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h7->x) * (static_cast<float>(mat_h7->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h7->y) * (static_cast<float>(mat_h7->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h8->x) * (static_cast<float>(mat_h8->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h8->y) * (static_cast<float>(mat_h8->getY()) - zero_point_f);
    }
  }

  sum *= scale_f;

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

__global__ void gemv_quantized_int4_multi_stage(uint4_2* mat, half* vec, half* mid_res,
                                unsigned int n, half scale, half zero_point, unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = blockIdx.x * (blockDim.x * num_per_thread) / 16 + threadIdx.x;
  uint4_2_4* mat4 = reinterpret_cast<uint4_2_4*>(mat);
  float4* vec4 = reinterpret_cast<float4*>(vec);

  float zero_point_f = static_cast<float>(zero_point);
  float scale_f = static_cast<float>(scale);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 4; iter++) {
    unsigned int j = 2 * (start_idx + iter * blockDim.x);
    if (j < n >> 3) {
      float4 vec_val_1 = vec4[j];  // 8 half
      float4 vec_val_2 = vec4[j + 1];
      const half2* vec_h1 = (half2*)&vec_val_1.x;
      const half2* vec_h2 = (half2*)&vec_val_1.y;
      const half2* vec_h3 = (half2*)&vec_val_1.z;
      const half2* vec_h4 = (half2*)&vec_val_1.w;
      const half2* vec_h5 = (half2*)&vec_val_2.x;
      const half2* vec_h6 = (half2*)&vec_val_2.y;
      const half2* vec_h7 = (half2*)&vec_val_2.z;
      const half2* vec_h8 = (half2*)&vec_val_2.w;

      uint4_2_4 mat_val_1 = mat4[row * (n >> 3) + j];
      uint4_2_4 mat_val_2 = mat4[row * (n >> 3) + j + 1];
      const uint4_2* mat_h1 = (uint4_2*)&mat_val_1.x;
      const uint4_2* mat_h2 = (uint4_2*)&mat_val_1.y;
      const uint4_2* mat_h3 = (uint4_2*)&mat_val_1.z;
      const uint4_2* mat_h4 = (uint4_2*)&mat_val_1.w;
      const uint4_2* mat_h5 = (uint4_2*)&mat_val_2.x;
      const uint4_2* mat_h6 = (uint4_2*)&mat_val_2.y;
      const uint4_2* mat_h7 = (uint4_2*)&mat_val_2.z;
      const uint4_2* mat_h8 = (uint4_2*)&mat_val_2.w;

      sum += static_cast<float>(vec_h1->x) * (static_cast<float>(mat_h1->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h1->y) * (static_cast<float>(mat_h1->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h2->x) * (static_cast<float>(mat_h2->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h2->y) * (static_cast<float>(mat_h2->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h3->x) * (static_cast<float>(mat_h3->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h3->y) * (static_cast<float>(mat_h3->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h4->x) * (static_cast<float>(mat_h4->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h4->y) * (static_cast<float>(mat_h4->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h5->x) * (static_cast<float>(mat_h5->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h5->y) * (static_cast<float>(mat_h5->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h6->x) * (static_cast<float>(mat_h6->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h6->y) * (static_cast<float>(mat_h6->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h7->x) * (static_cast<float>(mat_h7->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h7->y) * (static_cast<float>(mat_h7->getY()) - zero_point_f);
      sum += static_cast<float>(vec_h8->x) * (static_cast<float>(mat_h8->getX()) - zero_point_f);
      sum += static_cast<float>(vec_h8->y) * (static_cast<float>(mat_h8->getY()) - zero_point_f);
    }
  }

  sum *= scale_f;

  sum = warpReduceSum(sum, blockDim.x);

  if (blockDim.x <= WARP_SIZE) {
    if (tid == 0) {
      mid_res[row * gridDim.x + blockIdx.x] = __float2half(sum);
      // printf("mid_res[%d][%d] = %f\n", row, blockIdx.x, sum);
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

///////////////////////////// REDUCE SUM //////////////////////////////

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
