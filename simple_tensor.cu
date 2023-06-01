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