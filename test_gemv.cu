#include <curand.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <math.h>
#include <stdio.h>

#include <cassert>
#include <chrono>

#include "fast_gemv.cuh"
#include "fast_gemv_quantized.cuh"
#include "simple_tensor.h"

///////////////////////////// SOLVER (QUANTIZED) //////////////////////////////

static const half scale = 0.01;
static const half zero_point = 0.01;

SimpleTensor<half> solve_gemv_quantized_with_params(const SimpleTensor<int8_t>& mat, 
                                                    const SimpleTensor<half>& vec, 
                                                    unsigned int num_kernels, 
                                                    unsigned int block_dim_x,
                                                    unsigned int block_dim_y, 
                                                    unsigned int grid_dim_x) {
  assert(block_dim_y <= 32);
  unsigned int num_per_thread = mat.width_ / (block_dim_x * grid_dim_x);
  assert(num_per_thread >= 8);
  SimpleTensor<half> result(mat.height_, 1);
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
  // SimpleTensor<half> mid_result(mat.height_, grid_dim_x);
  // // launch kernel 1
  // dim3 grid_dim_1(grid_dim_x, mat.height_ / block_dim_y);  
  // dim3 block_dim_1(block_dim_x, block_dim_y);   
  // gemv_fp16_multi_stage<<<grid_dim_1, block_dim_1>>>(
  //     mat.data_, vec.data_, mid_result.data_, mat.width_, num_per_thread);
  // checkCudaErrors(cudaPeekAtLastError());
  // // launch kernel 2 (reduce)
  // dim3 grid_dim_2(1, mat.height_ / 32);  
  // dim3 block_dim_2(grid_dim_x, 32);
  // gemv_reduce_fp16<<<grid_dim_2, block_dim_2>>>(mid_result.data_,
  //                                               result.data_, grid_dim_x);
  // checkCudaErrors(cudaPeekAtLastError());
  return result;
}

void test_gemv_quantized_with_params(unsigned int size, unsigned int iter, unsigned int num_kernels, 
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
    res = solve_gemv_quantized_with_params(mat, vec, num_kernels, block_dim_x, block_dim_y, grid_dim_x);
  }

  // check correctness
  printf("checking...\n");
  int threads_per_block = 256;
  int num_blocks = (size + threads_per_block - 1) / threads_per_block;
  check_quantized_correctness<<<num_blocks, threads_per_block>>>(
      mat.device_data(), vec.device_data(), res.device_data(), scale, zero_point, size);
  printf("checked\n");
}


///////////////////////////// SOLVER //////////////////////////////

SimpleTensor<half> solve_gemv_with_params(const SimpleTensor<half>& mat, 
                                          const SimpleTensor<half>& vec, 
                                          unsigned int num_kernels, 
                                          unsigned int block_dim_x,
                                          unsigned int block_dim_y, 
                                          unsigned int grid_dim_x) {
  assert(block_dim_y <= 32);
  unsigned int num_per_thread = mat.width_ / (block_dim_x * grid_dim_x);
  assert(num_per_thread >= 8);
  SimpleTensor<half> result(mat.height_, 1);
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

///////////////////////////// UTILITIES //////////////////////////////

void print_cuda_info() {
  // for fun, just print out some stats on the machine

  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n",
           static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");
}
