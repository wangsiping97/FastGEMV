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

void test_gemv(unsigned int size, unsigned int iter) {
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
    res = mat.solve_gemv(vec);
  }

  // check correctness
  printf("checking...\n");
  int threads_per_block = 256;
  int num_blocks = (size + threads_per_block - 1) / threads_per_block;
  check_correctness<<<num_blocks, threads_per_block>>>(
      mat.device_data(), vec.device_data(), res.device_data(), size);
  printf("checked\n");
}
