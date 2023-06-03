#include <curand.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <math.h>
#include <stdio.h>

#include <cassert>
#include <chrono>

#include "utility.cuh"
#include "fast_gemv.cuh"
#include "simple_tensor.h"

///////////////////////////// SOLVER //////////////////////////////

static const half scale = 0.0625;
static const half zero_point = 0.01;

SimpleTensor<half> solve_gemv_int4_quantized_with_params(const SimpleTensor<uint4_2>& mat, 
                                                    const SimpleTensor<half>& vec, 
                                                    unsigned int num_kernels, 
                                                    unsigned int block_dim_x,
                                                    unsigned int block_dim_y, 
                                                    unsigned int grid_dim_x) {
  assert(mat.width_ * 2 == vec.height_);
  assert(block_dim_y <= 32);
  unsigned int num_per_thread = vec.height_ / (block_dim_x * grid_dim_x);
  assert(num_per_thread >= 16);
  SimpleTensor<half> result(vec.height_, 1);
  if (num_kernels == 1) {
    assert(grid_dim_x == 1);
    dim3 grid_dim(grid_dim_x, mat.height_ / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);
    gemv_quantized_int4_single_stage<<<grid_dim, block_dim>>>(mat.data_, vec.data_, result.data_, 
                                                              vec.height_, scale, zero_point, num_per_thread);
    checkCudaErrors(cudaPeekAtLastError());
    return result;
  }
  // num_kernels = 2
  assert(grid_dim_x > 1);
  SimpleTensor<half> mid_result(mat.height_, grid_dim_x);
  // launch kernel 1
  dim3 grid_dim_1(grid_dim_x, mat.height_ / block_dim_y);  
  dim3 block_dim_1(block_dim_x, block_dim_y);   
  gemv_quantized_int4_multi_stage<<<grid_dim_1, block_dim_1>>>(
      mat.data_, vec.data_, mid_result.data_, vec.height_, scale, zero_point, num_per_thread);
  checkCudaErrors(cudaPeekAtLastError());
  // launch kernel 2 (reduce)
  dim3 grid_dim_2(1, mat.height_ / 32);  
  dim3 block_dim_2(grid_dim_x, 32);
  gemv_reduce_fp16<<<grid_dim_2, block_dim_2>>>(mid_result.data_,
                                                result.data_, grid_dim_x);
  checkCudaErrors(cudaPeekAtLastError());
  return result;
}

SimpleTensor<half> solve_gemv_int8_quantized_with_params(const SimpleTensor<int8_t>& mat, 
                                                    const SimpleTensor<half>& vec, 
                                                    unsigned int num_kernels, 
                                                    unsigned int block_dim_x,
                                                    unsigned int block_dim_y, 
                                                    unsigned int grid_dim_x) {
  assert(mat.width_ == vec.height_);
  assert(block_dim_y <= 32);
  unsigned int num_per_thread = mat.width_ / (block_dim_x * grid_dim_x);
  assert(num_per_thread >= 8);
  SimpleTensor<half> result(vec.height_, 1);
  if (num_kernels == 1) {
    assert(grid_dim_x == 1);
    dim3 grid_dim(grid_dim_x, mat.height_ / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);
    gemv_quantized_int8_single_stage<<<grid_dim, block_dim>>>(mat.data_, vec.data_, result.data_, 
                                                              mat.width_, scale, zero_point, num_per_thread);
    checkCudaErrors(cudaPeekAtLastError());
    return result;
  }

  // num_kernels = 2
  assert(grid_dim_x > 1);
  SimpleTensor<half> mid_result(mat.height_, grid_dim_x);
  // launch kernel 1
  dim3 grid_dim_1(grid_dim_x, mat.height_ / block_dim_y);  
  dim3 block_dim_1(block_dim_x, block_dim_y);   
  gemv_quantized_int8_multi_stage<<<grid_dim_1, block_dim_1>>>(
      mat.data_, vec.data_, mid_result.data_, mat.width_, scale, zero_point, num_per_thread);
  checkCudaErrors(cudaPeekAtLastError());
  // launch kernel 2 (reduce)
  dim3 grid_dim_2(1, mat.height_ / 32);  
  dim3 block_dim_2(grid_dim_x, 32);
  gemv_reduce_fp16<<<grid_dim_2, block_dim_2>>>(mid_result.data_,
                                                result.data_, grid_dim_x);
  checkCudaErrors(cudaPeekAtLastError());
  return result;
}

SimpleTensor<half> solve_gemv_with_params(const SimpleTensor<half>& mat, 
                                          const SimpleTensor<half>& vec, 
                                          unsigned int num_kernels, 
                                          unsigned int block_dim_x,
                                          unsigned int block_dim_y, 
                                          unsigned int grid_dim_x) {
  assert(mat.width_ == vec.height_);
  assert(block_dim_y <= 32);
  unsigned int num_per_thread = mat.width_ / (block_dim_x * grid_dim_x);
  assert(num_per_thread >= 8);
  SimpleTensor<half> result(vec.height_, 1);
  if (num_kernels == 1) {
    assert(grid_dim_x == 1);
    dim3 grid_dim(grid_dim_x, mat.height_ / block_dim_y);
    dim3 block_dim(block_dim_x, block_dim_y);
    gemv_fp16_single_stage<<<grid_dim, block_dim>>>(mat.data_, vec.data_, result.data_,
                                           mat.width_, num_per_thread);
    checkCudaErrors(cudaPeekAtLastError());
    return result;
  }

  // num_kernels = 2
  assert(grid_dim_x > 1);
  SimpleTensor<half> mid_result(mat.height_, grid_dim_x);
  // launch kernel 1
  dim3 grid_dim_1(grid_dim_x, mat.height_ / block_dim_y);  
  dim3 block_dim_1(block_dim_x, block_dim_y);   
  gemv_fp16_multi_stage<<<grid_dim_1, block_dim_1>>>(
      mat.data_, vec.data_, mid_result.data_, mat.width_, num_per_thread);
  checkCudaErrors(cudaPeekAtLastError());
  // launch kernel 2 (reduce)
  dim3 grid_dim_2(1, mat.height_ / 32);  
  dim3 block_dim_2(grid_dim_x, 32);
  gemv_reduce_fp16<<<grid_dim_2, block_dim_2>>>(mid_result.data_,
                                                result.data_, grid_dim_x);
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
    half half_result = __float2half(result);
    float diff = __half2float(res[idx]) - __half2float(half_result);
    float delta = 0.125 * n / 512;
    if (diff > delta || diff < -delta) {
      printf("!!![idx=%d] %f != %f, diff=%f\n", idx, __half2float(res[idx]),
             __half2float(result), diff);
    }
  }
}

__global__ void check_int8_quantized_correctness(int8_t* mat, half* vec, half* res, half scale, half zero_point, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    float result = 0;
    for (int j = 0; j < n; ++j) {
      float dequantized_val = (static_cast<float>(mat[idx * n + j]) - static_cast<float>(zero_point)) * static_cast<float>(scale);
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

__global__ void check_int4_quantized_correctness(uint4_2* mat, half* vec, half* res, half scale, half zero_point, int mat_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < mat_size * 2) {
    float result = 0;
    for (int j = 0; j < mat_size; ++j) {
      uint8_t x = mat[idx * mat_size + j].getX();
      uint8_t y = mat[idx * mat_size + j].getY();
      float dequantized_x = (static_cast<float>(x) - static_cast<float>(zero_point)) * static_cast<float>(scale);
      float dequantized_y = (static_cast<float>(y) - static_cast<float>(zero_point)) * static_cast<float>(scale);
      result += dequantized_x * __half2float(vec[j * 2]);
      result += dequantized_y * __half2float(vec[j * 2 + 1]);
    }
    half half_result = __float2half(result);
    float diff = __half2float(res[idx]) - __half2float(half_result);
    float delta = 0.125 * mat_size / 256;
    if (diff > delta || diff < -delta) {
      printf("!!![idx=%d] %f != %f, diff=%f\n", idx, __half2float(res[idx]),
             __half2float(result), diff);
    }
  }
}

void test_gemv_int4_quantized_with_params(unsigned int size, unsigned int iter, unsigned int num_kernels, 
                           unsigned int block_dim_x, unsigned int block_dim_y, 
                           unsigned int grid_dim_x) {
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
    res = solve_gemv_int4_quantized_with_params(mat, vec, num_kernels, block_dim_x, block_dim_y, grid_dim_x);
  }

  // check correctness
  printf("checking...\n");
  int threads_per_block = 256;
  int num_blocks = (size + threads_per_block - 1) / threads_per_block;
  check_int4_quantized_correctness<<<num_blocks, threads_per_block>>>(
      mat.device_data(), vec.device_data(), res.device_data(), scale, zero_point, mat_width);
  printf("checked\n");
}

void test_gemv_int8_quantized_with_params(unsigned int size, unsigned int iter, unsigned int num_kernels, 
                           unsigned int block_dim_x, unsigned int block_dim_y, 
                           unsigned int grid_dim_x) {
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
    res = solve_gemv_int8_quantized_with_params(mat, vec, num_kernels, block_dim_x, block_dim_y, grid_dim_x);
  }

  // check correctness
  printf("checking...\n");
  int threads_per_block = 256;
  int num_blocks = (size + threads_per_block - 1) / threads_per_block;
  check_int8_quantized_correctness<<<num_blocks, threads_per_block>>>(
      mat.device_data(), vec.device_data(), res.device_data(), scale, zero_point, size);
  printf("checked\n");
}

void test_gemv_with_params(unsigned int size, unsigned int iter, unsigned int num_kernels, 
                           unsigned int block_dim_x, unsigned int block_dim_y, 
                           unsigned int grid_dim_x) {
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
    res = solve_gemv_with_params(mat, vec, num_kernels, block_dim_x, block_dim_y, grid_dim_x);
  }

  // check correctness
  printf("checking...\n");
  int threads_per_block = 256;
  int num_blocks = (size + threads_per_block - 1) / threads_per_block;
  check_correctness<<<num_blocks, threads_per_block>>>(
      mat.device_data(), vec.device_data(), res.device_data(), size);
  printf("checked\n");
}
