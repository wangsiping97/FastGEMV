#include <curand.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <math.h>
#include <stdio.h>

#include <cassert>
#include <chrono>

#include "fast_gemv.cuh"
#include "simple_tensor.h"
#include "utility.cuh"

///////////////////////////// SOLVER //////////////////////////////

SimpleTensor<half> solve_gemv_int4_quantized_with_params(
    const SimpleTensor<uint4_2>& mat, const SimpleTensor<half>& vec,
    unsigned int block_dim_x, unsigned int block_dim_y, float scale_f,
    float zero_point_f) {
  half scale = __float2half(scale_f);
  half zero_point = __float2half(zero_point_f);
  assert(mat.width_ * 2 == vec.height_);
  assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = vec.height_ / block_dim_x;
  assert(num_per_thread >= 16);
  SimpleTensor<half> result(vec.height_, 1);
  dim3 grid_dim(1, mat.height_ / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  gemv_quantized_int4<<<grid_dim, block_dim>>>(mat.data_, vec.data_,
                                               result.data_, vec.height_, scale,
                                               zero_point, num_per_thread);
  checkCudaErrors(cudaPeekAtLastError());
  return result;
}

SimpleTensor<half> solve_gemv_int8_quantized_with_params(
    const SimpleTensor<int8_t>& mat, const SimpleTensor<half>& vec,
    unsigned int block_dim_x, unsigned int block_dim_y, float scale_f,
    float zero_point_f) {
  half scale = __float2half(scale_f);
  half zero_point = __float2half(zero_point_f);
  assert(mat.width_ == vec.height_);
  assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = mat.width_ / block_dim_x;
  assert(num_per_thread >= 8);
  SimpleTensor<half> result(vec.height_, 1);
  dim3 grid_dim(1, mat.height_ / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  gemv_quantized_int8<<<grid_dim, block_dim>>>(mat.data_, vec.data_,
                                               result.data_, mat.width_, scale,
                                               zero_point, num_per_thread);
  checkCudaErrors(cudaPeekAtLastError());
  return result;
}

SimpleTensor<half> solve_gemv_with_params(const SimpleTensor<half>& mat,
                                          const SimpleTensor<half>& vec,
                                          unsigned int block_dim_x,
                                          unsigned int block_dim_y) {
  assert(mat.width_ == vec.height_);
  assert(block_dim_y <= SHARED_MEM_MAX_ROWS);
  assert(block_dim_x * block_dim_y <= MAX_THREADS_PER_BLOCK);
  unsigned int num_per_thread = mat.width_ / block_dim_x;
  assert(num_per_thread >= 8);
  SimpleTensor<half> result(vec.height_, 1);
  dim3 grid_dim(1, mat.height_ / block_dim_y);
  dim3 block_dim(block_dim_x, block_dim_y);
  gemv_fp16<<<grid_dim, block_dim>>>(mat.data_, vec.data_, result.data_,
                                     mat.width_, num_per_thread);
  checkCudaErrors(cudaPeekAtLastError());
  return result;
}

///////////////////////////// TEST //////////////////////////////

__global__ void check_correctness(half* mat, half* vec, half* res, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float result = 0;
    for (int j = 0; j < n; ++j) {
      result += __half2float(mat[idx * n + j]) * __half2float(vec[j]);
    }
    float diff = result - __half2float(res[idx]);
    float delta = 0.125 * n / 512;
    if (diff > delta || diff < -delta) {
      printf("!!![idx=%d] %f != %f, diff=%f\n", idx, __half2float(res[idx]),
             result, diff);
    }
  }
}

__global__ void check_int8_quantized_correctness(int8_t* mat, half* vec,
                                                 half* res, half scale,
                                                 half zero_point, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float result = 0;
    for (int j = 0; j < n; ++j) {
      float dequantized_val = (static_cast<float>(mat[idx * n + j]) -
                               static_cast<float>(zero_point)) *
                              static_cast<float>(scale);
      result += dequantized_val * __half2float(vec[j]);
    }
    float diff = result - __half2float(res[idx]);
    float delta = 0.125 * n / 512;
    if (diff > delta || diff < -delta) {
      printf("!!![idx=%d] %f != %f, diff=%f\n", idx, __half2float(res[idx]),
             result, diff);
    }
  }
}

__global__ void check_int4_quantized_correctness(uint4_2* mat, half* vec,
                                                 half* res, half scale,
                                                 half zero_point,
                                                 int mat_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < mat_size * 2) {
    float result = 0;
    for (int j = 0; j < mat_size; ++j) {
      uint8_t x = mat[idx * mat_size + j].getX();
      uint8_t y = mat[idx * mat_size + j].getY();
      float dequantized_x =
          (static_cast<float>(x) - static_cast<float>(zero_point)) *
          static_cast<float>(scale);
      float dequantized_y =
          (static_cast<float>(y) - static_cast<float>(zero_point)) *
          static_cast<float>(scale);
      result += dequantized_x * __half2float(vec[j * 2]);
      result += dequantized_y * __half2float(vec[j * 2 + 1]);
    }
    float diff = result - __half2float(res[idx]);
    float delta = 0.125 * mat_size / 256;
    if (diff > delta || diff < -delta) {
      printf("!!![idx=%d] %f != %f, diff=%f\n", idx, __half2float(res[idx]),
             result, diff);
    }
  }
}

void test_gemv_int4_quantized_with_params(unsigned int size, unsigned int iter,
                                          unsigned int block_dim_x,
                                          unsigned int block_dim_y, float scale,
                                          float zero_point) {
  cudaSetDevice(0);
  // generate data
  const unsigned int mat_width = size / 2;
  SimpleTensor<uint4_2> mat(size, mat_width);
  SimpleTensor<half> vec(size, 1);
  mat.reset();
  vec.reset();

  // compute dot product
  printf("solving...\n");
  SimpleTensor<half> res(size, 1);
  for (int i = 0; i < iter; ++i) {
    res = solve_gemv_int4_quantized_with_params(mat, vec, block_dim_x,
                                                block_dim_y, scale, zero_point);
  }

  // check correctness
  printf("checking...\n");
  int threads_per_block = 256;
  int num_blocks = (size + threads_per_block - 1) / threads_per_block;
  check_int4_quantized_correctness<<<num_blocks, threads_per_block>>>(
      mat.device_data(), vec.device_data(), res.device_data(), scale,
      zero_point, mat_width);
  printf("checked\n");
}

void test_gemv_int8_quantized_with_params(unsigned int size, unsigned int iter,
                                          unsigned int block_dim_x,
                                          unsigned int block_dim_y, float scale,
                                          float zero_point) {
  cudaSetDevice(0);
  // generate data
  SimpleTensor<int8_t> mat(size, size);
  SimpleTensor<half> vec(size, 1);
  mat.reset();
  vec.reset();

  // compute the dot product
  printf("solving...\n");
  SimpleTensor<half> res(size, 1);

  for (int i = 0; i < iter; ++i) {
    res = solve_gemv_int8_quantized_with_params(mat, vec, block_dim_x,
                                                block_dim_y, scale, zero_point);
  }

  // check correctness
  printf("checking...\n");
  int threads_per_block = 256;
  int num_blocks = (size + threads_per_block - 1) / threads_per_block;
  check_int8_quantized_correctness<<<num_blocks, threads_per_block>>>(
      mat.device_data(), vec.device_data(), res.device_data(), scale,
      zero_point, size);
  printf("checked\n");
}

void test_gemv_with_params(unsigned int size, unsigned int iter,
                           unsigned int block_dim_x, unsigned int block_dim_y) {
  cudaSetDevice(0);
  // generate data
  SimpleTensor<half> mat(size, size);
  SimpleTensor<half> vec(size, 1);
  mat.reset();
  vec.reset();

  // compute the dot product
  printf("solving...\n");
  SimpleTensor<half> res(size, 1);

  for (int i = 0; i < iter; ++i) {
    res = solve_gemv_with_params(mat, vec, block_dim_x, block_dim_y);
  }

  // check correctness
  printf("checking...\n");
  int threads_per_block = 256;
  int num_blocks = (size + threads_per_block - 1) / threads_per_block;
  check_correctness<<<num_blocks, threads_per_block>>>(
      mat.device_data(), vec.device_data(), res.device_data(), size);
  printf("checked\n");
}
