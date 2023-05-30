#include <cassert>

#include "fast_gemv.cuh"
#include "simple_tensor.h"

const unsigned int THREAD_PER_BLOCK = 256;

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

SimpleTensor SimpleTensor::solve_gemv(const SimpleTensor& other) const {
  assert(width_ == other.height_);
  assert(other.width_ == 1);
  SimpleTensor result(height_, 1);

  if (width_ <= 512) {
    const unsigned int block_num = 1;
    const unsigned int thread_per_block = 32;
    unsigned int num_per_thread = width_ / (thread_per_block * block_num);
    if (num_per_thread == 0) {
      num_per_thread = 1;
    }
    dim3 grid_dim(block_num, height_);
    dim3 block_dim(thread_per_block, 1);
    gemv_fp16_512<<<grid_dim, block_dim>>>(data_, other.data_, result.data_,
                                           width_, thread_per_block,
                                           num_per_thread);
    checkCudaErrors(cudaPeekAtLastError());
    return result;
  }

  const unsigned int block_num = 32;
  unsigned int num_per_thread = height_ / (THREAD_PER_BLOCK * block_num);
  if (num_per_thread == 0) num_per_thread = 1;

  SimpleTensor mid_result(height_, block_num);
  // launch naive kernel
  // TODO: optimize

  dim3 grid_dim(block_num, height_);
  dim3 block_dim(THREAD_PER_BLOCK, 1);
  gemv_fp16<<<grid_dim, block_dim>>>(data_, other.data_, mid_result.data_,
                                     width_, THREAD_PER_BLOCK, num_per_thread);
  checkCudaErrors(cudaPeekAtLastError());
  dim3 grid_dim_reduce(1, height_);
  dim3 block_dim_reduce(block_num, 1);
  gemv_reduce_fp16<<<grid_dim_reduce, block_dim_reduce>>>(
      mid_result.data_, result.data_, block_num);
  checkCudaErrors(cudaPeekAtLastError());
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