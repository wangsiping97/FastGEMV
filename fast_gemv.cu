#include <stdio.h>
#include <ctime>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <cuda_fp16.h>

// Define the error checking function
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__ )

void check(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error = %d at %s:%d '%s'\n", static_cast<unsigned int>(result), file, line, func);
        exit(1);
    }
}

void print_cuda_info() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
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

__global__ void generate_random_numbers(half* numbers, int Np) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < Np) {
        curandState state;
        curand_init(clock64(), i, 0, &state);
        numbers[i] = __float2half(curand_uniform(&state));
    }
}

void gen_matrix(unsigned int size) {
    cudaSetDevice(0);
    half* numbers;

    unsigned int total_elements = size * size;

    // allocate memory on the device
    checkCudaErrors(cudaMalloc((void**)&numbers, total_elements * sizeof(half)));

    // initialize the states
    generate_random_numbers<<<(total_elements + 255) / 256, 256>>>(numbers, total_elements);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // allocate memory on the host
    half* host_numbers = (half*)malloc(total_elements * sizeof(half));

    // copy the numbers back to the host
    checkCudaErrors(cudaMemcpy(host_numbers, numbers, total_elements * sizeof(half), cudaMemcpyDeviceToHost));

    // print out the first 10 * 10 numbers
    for (unsigned int i = 0; i < 10 && i < size; ++i) {
        for (unsigned int j = 0; j < 10 && j < size; ++j) {
            printf("%f ", __half2float(host_numbers[i * size + j]));
        }
        printf("\n");
    }

    // free the memory we allocated on the GPU
    checkCudaErrors(cudaFree(numbers));

    // free the memory we allocated on the CPU
    free(host_numbers);
}
