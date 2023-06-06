#ifndef UTILITY_H_
#define UTILITY_H_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdint>

///////////////////////////// DATA TYPES //////////////////////////////

struct uint4_2 {
  uint8_t data;

  uint4_2(uint8_t x = 0, uint8_t y = 0) {
    setX(x);
    setY(y);
  }

  __host__ __device__ uint8_t getX() const {
    return data & 0x0F;  // get the lower 4 bits
  }

  __host__ __device__ uint8_t getY() const {
    return (data >> 4) & 0x0F;  // get the upper 4 bits
  }

  __host__ __device__ void setX(uint8_t x) {
    data = (data & 0xF0) | (x & 0x0F);  // set the lower 4 bits
  }

  __host__ __device__ void setY(uint8_t y) {
    data = (data & 0x0F) | ((y & 0x0F) << 4);  // set the upper 4 bits
  }
};

struct half4 {
  half x, y, z, w;
};
struct int8_2 {
  int8_t x, y;
};
struct uint4_2_4 {
  uint4_2 x, y, z, w;
};

///////////////////////////// CUDA UTILITIES //////////////////////////////

void print_cuda_info();

// Define the error checking function
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t result, char const* const func, const char* const file,
           int const line);

__global__ void generate_random_numbers(half* numbers, int Np);
__global__ void generate_random_int8_numbers(int8_t* numbers, int Np);
__global__ void generate_random_int4_numbers(uint4_2* numbers, int Np);

#endif  // UTILITY_H_