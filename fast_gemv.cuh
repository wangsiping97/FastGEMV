#ifndef FAST_GEMV_CUH_
#define FAST_GEMV_CUH_

__global__ void gemv_fp16_128(half* mat, half* vec, half* res, int n);

#endif  // FAST_GEMV_CUH_