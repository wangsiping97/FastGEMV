#include <curand.h>
#include <curand_kernel.h>
#include <driver_functions.h>
#include <math.h>
#include <stdio.h>

#include <cassert>
#include <chrono>

#include "fast_gemv.cuh"
#include "simple_tensor.h"

///////////////////////////// SOLVER //////////////////////////////

void test_gemv_with_params(unsigned int size, unsigned int iter, unsigned int num_kernels, 
                           unsigned int block_dim_x, unsigned int block_dim_y, 
                           unsigned int grid_dim_x) {
  cudaSetDevice(0);
  // generate data
  SimpleTensor mat = SimpleTensor(size, size);
  SimpleTensor vec = SimpleTensor(size, 1);
  mat.reset();
  vec.reset();

  // compute the dot product
  printf("solving...\n");
  SimpleTensor res = SimpleTensor(size, 1);

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
