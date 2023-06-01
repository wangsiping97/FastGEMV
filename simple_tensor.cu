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

SimpleTensor solve_gemv_with_params(const SimpleTensor& mat, const SimpleTensor& vec, 
                                    unsigned int num_kernels, unsigned int block_dim_x,
                                    unsigned int block_dim_y, unsigned int grid_dim_x) {
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