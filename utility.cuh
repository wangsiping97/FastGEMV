#ifndef UTILITY_H_
#define UTILITY_H_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdio.h>

///////////////////////////// DATA TYPES //////////////////////////////

struct half4 { half x, y, z, w; };
struct int8_2 { int8_t x, y; };
struct int4_2 {
  int x: 4;
  int y: 4;
};

// class uint4_t {
// public: 
//   uint4_t(int value)
//     : storage(reinterpret_cast<uint8_t const &>(value) & kMask) {}
  
//   bool operator==(uint4_t const &rhs) const {
//     return storage == rhs.storage;
//   }

//   bool operator!=(uint4_t const &rhs) const {
//     return storage != rhs.storage;
//   }

//   bool operator<=(uint4_t const &rhs) const {
//     return storage < rhs.storage;
//   }

//   bool operator<(uint4_t const &rhs) const {
//     return storage < rhs.storage;
//   }

//   bool operator>=(uint4_t const &rhs) const {
//     return !(*this < rhs);
//   }

//   bool operator>(uint4_t const &rhs) const {
//     return !(*this <= rhs);
//   }

//   uint8_t value() {
//     return storage;
//   }

// private:
//   /// Number of bits
//   static int const kBits = 4;
//   /// Bitmask used to truncate from larger integers
//   static uint8_t const kMask = uint8_t((1 << kBits) - 1);
//   /// Data member
//   uint8_t storage;
// };

///////////////////////////// CUDA UTILITIES //////////////////////////////

void print_cuda_info();

// Define the error checking function
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t result, char const* const func, const char* const file,
           int const line);

__global__ void generate_numbers(half* numbers, int Np);
__global__ void generate_random_numbers(half* numbers, int Np);
__global__ void generate_random_int8_numbers(int8_t* numbers, int Np);

#endif // UTILITY_H_