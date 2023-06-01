#include <cassert>

#include "fast_gemv.cuh"
#include "simple_tensor.h"

void check(cudaError_t result, char const* const func, const char* const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error = %s at %s:%d '%s'\n",
            cudaGetErrorString(result), file, line, func);
    exit(1);
  }
}

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

SimpleTensor solve_gemv_with_params(const SimpleTensor& mat, const SimpleTensor& vec, int num_kernels, int block_dim_x, int block_dim_y, int grid_dim_x) {
  assert(block_dim_y <= 32);
  unsigned int num_per_thread = mat.width_ / (block_dim_x * grid_dim_x);
  assert(num_per_thread >= 8);
  SimpleTensor result(mat.height_, 1);
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
  SimpleTensor mid_result(mat.height_, grid_dim_x);
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

SimpleTensor SimpleTensor::solve_gemv(const SimpleTensor& other) const {
  assert(width_ == other.height_);
  assert(other.width_ == 1);
  SimpleTensor result(height_, 1);

  if (width_ <= 2048) {
    const unsigned int block_per_row = 1;
    const unsigned int thread_per_block = 32;
    unsigned int num_per_thread = width_ / (thread_per_block * block_per_row);
    if (num_per_thread == 0) {
      num_per_thread = 1;
    }
    dim3 grid_dim(block_per_row, height_ / 4);  // 1 * 128 blocks
    dim3 block_dim(thread_per_block, 4);        // 32 * 4 threads
    gemv_fp16_single_stage<<<grid_dim, block_dim>>>(data_, other.data_, result.data_,
                                           width_, num_per_thread);
    checkCudaErrors(cudaPeekAtLastError());
    return result;
  }

  if (width_ == 4096) {
    const unsigned int block_per_row = 8;
    const unsigned int thread_per_block = 32;
    unsigned int num_per_thread = width_ / (thread_per_block * block_per_row);
    SimpleTensor mid_result(height_, block_per_row);
    // launch kernel 1
    dim3 grid_dim_1(block_per_row, height_ / 4);  // 8 * 1024 blocks
    dim3 block_dim_1(thread_per_block, 4);        // 32 * 4 threads
    gemv_fp16_multi_stage<<<grid_dim_1, block_dim_1>>>(
        data_, other.data_, mid_result.data_, width_, num_per_thread);
    checkCudaErrors(cudaPeekAtLastError());
    // launch kernel 2 (reduce)
    dim3 grid_dim_2(1, height_ / 32);     // 1 * 128 blocks
    dim3 block_dim_2(block_per_row, 32);  // 8 * 32 threads
    gemv_reduce_fp16<<<grid_dim_2, block_dim_2>>>(mid_result.data_,
                                                  result.data_, block_per_row);
    checkCudaErrors(cudaPeekAtLastError());
    return result;
  }

  if (width_ == 8192) {
    const unsigned int block_per_row = 8;
    const unsigned int thread_per_block = 32;
    unsigned int num_per_thread = width_ / (thread_per_block * block_per_row);
    SimpleTensor mid_result(height_, block_per_row);
    // launch kernel 1
    dim3 grid_dim_1(block_per_row, height_ / 8);  // 16 * 4096 blocks
    dim3 block_dim_1(thread_per_block, 8);        // 32 * 2 threads
    gemv_fp16_multi_stage<<<grid_dim_1, block_dim_1>>>(
        data_, other.data_, mid_result.data_, width_, num_per_thread);
    checkCudaErrors(cudaPeekAtLastError());
    // launch kernel 2 (reduce)
    dim3 grid_dim_2(1, height_ / 32);     // 1 * 256 blocks
    dim3 block_dim_2(block_per_row, 32);  // 32 * 32 threads
    gemv_reduce_fp16<<<grid_dim_2, block_dim_2>>>(mid_result.data_,
                                                  result.data_, block_per_row);
    checkCudaErrors(cudaPeekAtLastError());
    return result;
  }

  if (width_ >= 16384) {
    const unsigned int block_per_row = 4;
    const unsigned int thread_per_block = 32;
    unsigned int num_per_thread = width_ / (thread_per_block * block_per_row);
    SimpleTensor mid_result(height_, block_per_row);
    // launch kernel 1
    dim3 grid_dim_1(block_per_row, height_ / 8);  // 4 * 2048 blocks
    dim3 block_dim_1(thread_per_block, 8);        // 32 * 8 threads
    gemv_fp16_multi_stage<<<grid_dim_1, block_dim_1>>>(
        data_, other.data_, mid_result.data_, width_, num_per_thread);
    checkCudaErrors(cudaPeekAtLastError());
    // launch kernel 2 (reduce)
    dim3 grid_dim_2(1, height_ / 64);     // 1 * 256 blocks
    dim3 block_dim_2(block_per_row, 64);  // 32 * 64 threads
    gemv_reduce_fp16<<<grid_dim_2, block_dim_2>>>(mid_result.data_,
                                                  result.data_, block_per_row);
    checkCudaErrors(cudaPeekAtLastError());
    return result;
  }

  const unsigned int block_num = 1;
  const unsigned int THREAD_PER_BLOCK = 64;
  unsigned int num_per_thread = width_ / (THREAD_PER_BLOCK * block_num);
  if (num_per_thread == 0) num_per_thread = 1;

  SimpleTensor mid_result(height_, block_num);

  unsigned int blocks_y = height_ / 4;
  dim3 grid_dim(block_num, blocks_y);
  dim3 block_dim(THREAD_PER_BLOCK, height_ / blocks_y);
  gemv_fp16<<<grid_dim, block_dim>>>(data_, other.data_, mid_result.data_,
                                     width_, THREAD_PER_BLOCK, num_per_thread);
  checkCudaErrors(cudaPeekAtLastError());
  blocks_y = height_ / 32;
  dim3 grid_dim_reduce(1, blocks_y);
  dim3 block_dim_reduce(block_num, height_ / blocks_y);
  gemv_reduce_fp16<<<grid_dim_reduce, block_dim_reduce>>>(
      mid_result.data_, result.data_, block_num);
  checkCudaErrors(cudaPeekAtLastError());
  
  // launch naive kernel
  // int threads_per_block = 10;
  // int num_blocks = (width_ + threads_per_block - 1) / threads_per_block;
  // gemv_naive<<<num_blocks, threads_per_block>>>(data_, other.data_,
  //                                               result.data_, width_);

  return result;
}

void SimpleTensor::reset() {
  unsigned int total_elements = height_ * width_;
  int threads_per_block = 256;
  int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
  generate_random_numbers<<<num_blocks, threads_per_block>>>(data_,
                                                             total_elements);
  checkCudaErrors(cudaPeekAtLastError());
}

void SimpleTensor::to_host(half* host_data, unsigned n) {
  unsigned int total_elements = height_ * width_;
  assert(n <= total_elements);
  checkCudaErrors(
      cudaMemcpy(host_data, data_, n * sizeof(half), cudaMemcpyDeviceToHost));
}