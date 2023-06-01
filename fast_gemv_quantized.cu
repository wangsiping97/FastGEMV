#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <stdio.h>

#include "fast_gemv_quantized.cuh"

#define WARP_SIZE 32

struct half4 { half x, y, z, w; };
struct uint8_2 { uint8_t x, y; };
struct float4_2 { float4 x, y; };

// each thread computes 8 * 2(row) results
// gridDim.y = 128, blockDim.y = 2
__global__ void init_table_int8(half* vec, float* table, unsigned int n, float scale, int16_t zero_point) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int start_row = blockIdx.y * blockDim.y + threadIdx.y;
  float4* vec4 = reinterpret_cast<float4*>(vec);
  float4_2* table8 = reinterpret_cast<float4_2*>(table);
  if (idx >= n >> 3) {
    return;
  }
  float4 vec_val = vec4[idx];
  const half2* vec_h1 = (half2*)&vec_val.x;
  const half2* vec_h2 = (half2*)&vec_val.y;
  const half2* vec_h3 = (half2*)&vec_val.z;
  const half2* vec_h4 = (half2*)&vec_val.w;
  for (uint8_t i = start_row; i < start_row + blockDim.y; ++i) {
    float4_2 res;
    int8_t val = (int8_t)i;
    res.x.x = (scale * (val - zero_point)) * static_cast<float>(vec_h1->x);
    res.x.y = (scale * (val - zero_point)) * static_cast<float>(vec_h1->y);
    res.x.z = (scale * (val - zero_point)) * static_cast<float>(vec_h2->x);
    res.x.w = (scale * (val - zero_point)) * static_cast<float>(vec_h2->y);
    res.y.x = (scale * (val - zero_point)) * static_cast<float>(vec_h3->x);
    res.y.y = (scale * (val - zero_point)) * static_cast<float>(vec_h3->y);
    res.y.z = (scale * (val - zero_point)) * static_cast<float>(vec_h4->x);
    res.y.w = (scale * (val - zero_point)) * static_cast<float>(vec_h4->y);
    // write res to table
    table8[i * n / 8 + idx] = res;
  }
}

// num_per_thread >= 8
__global__ void gemv_quantized_int8_single_stage(int8_t* mat, half* res, float* table, unsigned int n,
                              unsigned int num_per_thread) {
  float sum = 0;
  // each thread load num_per_thread elements from global
  unsigned int tid = threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int start_idx = threadIdx.x;
  half4* mat4 = reinterpret_cast<half4*>(mat);

#pragma unroll
  for (int iter = 0; iter < num_per_thread >> 3; iter++) {
    unsigned int j = start_idx + iter * blockDim.x;
    if (j < n >> 3) {
      half4 mat_val = mat4[row * (n >> 3) + j];
      const uint8_2* mat_h1 = (uint8_2*)&mat_val.x;
      const uint8_2* mat_h2 = (uint8_2*)&mat_val.y;
      const uint8_2* mat_h3 = (uint8_2*)&mat_val.z;
      const uint8_2* mat_h4 = (uint8_2*)&mat_val.w;
      sum += table[mat_h1->x * n + j * 8];
      sum += table[mat_h1->y * n + j * 8 + 1];
      sum += table[mat_h2->x * n + j * 8 + 2];
      sum += table[mat_h2->y * n + j * 8 + 3];
      sum += table[mat_h3->x * n + j * 8 + 4];
      sum += table[mat_h3->y * n + j * 8 + 5];
      sum += table[mat_h4->x * n + j * 8 + 6];
      sum += table[mat_h4->y * n + j * 8 + 7];
    }
  }

  sum = warpReduceSum2(sum, blockDim.x);

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
  if (warpId == 0) sum = warpReduceSum2(sum, blockDim.x / WARP_SIZE);
  if (tid == 0) {
    res[row] = __float2half(sum);
  }
}

///////////////////////////// UTILITIES //////////////////////////////

__device__ __forceinline__ float warpReduceSum2(float sum,
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

__global__ void generate_random_int8_numbers(int8_t* numbers, int Np) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < Np) {
    curandState state;
    curand_init(clock64(), i, 0, &state);
    numbers[i] = static_cast<int8_t>(curand(&state) % 128); // Random int8 number [-128, 127]
  }
}

__global__ void check_quantized_correctness(int8_t* mat, half* vec, half* res, float scale, int16_t zero_point, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float result = 0;
    for (int j = 0; j < n; ++j) {
      float dequantized_val = (mat[idx * n + j] - zero_point) * scale;
      result += dequantized_val * __half2float(vec[j]);
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
