#include <curand_kernel.h>
#include <driver_functions.h>

#include "utility.cuh"

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

void check(cudaError_t result, char const* const func, const char* const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error = %s at %s:%d '%s'\n",
            cudaGetErrorString(result), file, line, func);
    exit(1);
  }
}

__global__ void generate_random_numbers(half* numbers, int Np) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < Np) {
    curandState state;
    curand_init(clock64(), i, 0, &state);
    numbers[i] = __float2half(curand_uniform(&state));
  }
}

__global__ void generate_numbers(half* numbers, int Np) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < Np) {
    numbers[i] = __float2half(i / 1000.0);
  }
}

__global__ void generate_random_int8_numbers(int8_t* numbers, int Np) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < Np) {
    curandState state;
    curand_init(clock64(), i, 0, &state);
    numbers[i] = static_cast<int8_t>(curand(&state) % 256 - 128); // Random int8 number [-128, 127]
  }
}