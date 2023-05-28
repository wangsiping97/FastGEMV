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

SimpleTensor SimpleTensor::solve_gemv(const SimpleTensor& other) const {
  assert(width_ == other.height_);
  assert(other.width_ == 1);
  SimpleTensor result(height_, 1);
  // // start timing
  // cudaEvent_t startEvent, stopEvent;
  // cudaEventCreate(&startEvent);
  // cudaEventCreate(&stopEvent);
  // cudaEventRecord(startEvent, 0);
  // launch naive kernel
  // TODO: optimize
  int threads_per_block = 256;
  int num_blocks = (width_ + threads_per_block - 1) / threads_per_block;
  gemv_naive<<<num_blocks, threads_per_block>>>(data_, other.data_,
                                                result.data_, width_);
  // // Record the stop event
  // cudaEventRecord(stopEvent, 0);
  // cudaEventSynchronize(stopEvent);
  // float elapsedTime;
  // cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
  // printf("Kernel time: %.3f µs\n", elapsedTime * 1000);
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