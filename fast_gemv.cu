#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <driver_functions.h>
#include <curand_kernel.h>

#include "fast_gemv.cuh"

__global__ void gemv_fp16_128(half* mat, half* vec, half* res, int n) {

}