# Method and Result

## Optimization Strategy

### P threads per dot product

Instead of having one thread to produce 1 component of the result vector by computing the dot product between each row of the matrix and the vector, we had `p` threads to be responsible for one row, and each thread is responsible for `size / p` elements both in the matrix and in the vector.

This is the big picture of our algorithm: 

![1](./pics/1.png)

As shown in the above image, in each row, `p` threads (here `p`=8) compute one section of the dot product, and then the partial results are added together to obtain one value of `result[i]`. 

Hence, the problem can be divided into 2 parts: 

- How to optimize the computation time of generating the partial result in a single thread?
- How to optimize the addition of each partial results in the same row? 

### Vectorization

#### fp16

#### Quantized int8

#### Quantized int4

### Reduce sum

When adding partial results computed by each thread, instead of summing up the values one by one, we applied similar ideas to CUTLASS's [warp and block reduction](https://github.com/NVIDIA/cutlass/blob/main/tools/util/include/cutlass/util/device_utils.h). Specifically, we have a self-implemented `warpReduceSum()` like this: 

```c++
__device__ __forceinline__ float warpReduceSum(float sum,
                                               unsigned int p) {
  if (p >= 32)
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
  if (p >= 16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
  if (p >= 8)
    sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
  if (p >= 4)
    sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
  if (p >= 2)
    sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
  return sum;
}
```

In this function, each thread starts with a sum, and the goal is to compute the total sum across all threads in the same warp and distribute the result back to all threads in that warp. The function achieves this with several rounds of pairwise summing and shifting.

The `__shfl_down_sync(mask, var, delta)` function is used to perform the sum reduction. It takes a mask (`0xffffffff`, representing all threads in a warp) which indicates the threads participating in the operation, var which is the value to be shifted, and delta which is the number of positions to shift. This function returns the value of var held by the thread delta positions below the calling thread in the warp. If the target thread is out of bounds, the calling thread's own value is returned.

The workflow can be illustrated as below: 

![2](./pics/2.png)

When `p` is less than `WARP_SIZE` (32), only 1 warp exists, the return value of `warpReduceSum()` would be the total sum across all threads in the warp, which will be directly written to the `result[i]`.

When `p` is greater than `WARP_SIZE`, assuming there are `m` warps (`m` <= `WARP SIZE`), then there are 2 steps: 
  - step 1: Each warp computes the sum of values in itself by calling `warpReduceSum()`.
  - step 2: Each warp `j` loads its sum, `sum_j`, to the shared memory. The first `m` threads in the `warp_0` reads the values from the shared memory, and perform a reduction sum through `warpReduceSum()`. 

### GPU thread layout

### Other attempted methods

#### Shared memory

#### 2 kernels

#### Hash table (quantized int8)

## Result

### Bandwidth estimation method

The formula we used to estimate the achieved bandwidth is: 

```
Estimated BW = (sizeof(mat) + sizeof(vec) + sizeof(result)) / T 
```

This is because we need to read each data of both the matrix and the vector once, and then write it to the result vector. More specifically, 

- when matrix is in fp16, BW = (2 * n^2 + 2 * 2n) / T (GB/s)

- when matrix is in int8, BW = (n^2 + 2 * 2n) / T (GB/s)

- when matrix is in int4, BW = (0.5 * n^2 + 2 * 2n) / T (GB/s)

where n is the matrix size and T is in ns. 

### On P100

We only tested the runtime and estimated bandwidth for non-quantized fp16 matrices. Here is the result compared with pytorch:

Total GEMV kernel(s) average runtime (ns): 

| Size  | Pytorch   | My Kernel | Speedup |
| ----- | --------- | --------- | ------- |
| 512   | 5891.6    | 4087.6    | 1.441   |
| 1024  | 9047.2    | 5559.0    | 1.627   |
| 2048  | 28231.4   | 18197.8   | 1.551   |
| 4096  | 100727.5  | 62975.1   | 1.599   |
| 8192  | 618377.7  | 229488.3  | 2.695   |
| 16384 | 1588132.0 | 891879.9  | 1.781   |

Estimated BW (Max: 732 GB/s):

| Size  | Pytorch | My Kernel |
| ----- | ------- | --------- |
| 512   | 89.337  | 128.764   |
| 1024  | 232.254 | 377.990   |
| 2048  | 297.428 | 461.418   |
| 4096  | 333.284 | 533.081   |
| 8192  | 202.083 | 584.999   |
| 16384 | 320.796 | 602.028   |

Here are the parameters used for above results: 

| size  | blockDim.x | blockDim.y | gridDim.x | gridDim.y |
| ----- | ---------- | ---------- | --------- | --------- |
| 512   | 32         | 4          | 1         | 128       |
| 1024  | 32         | 4          | 1         | 256       |
| 2048  | 32         | 4          | 1         | 512       |
| 4096  | 128        | 8          | 1         | 512       |
| 8192  | 256        | 4          | 1         | 2048      |
| 16384 | 512        | 2          | 1         | 8192      |

### On 3090

Here is the overall runtime and achieved bandwidth compared with pytorch. Note that pytorch doesn't support direct computation of quantized GEMV.

Total GEMV kernel(s) average runtime (ns): 

| Size  | Pytorch  | My Kernel | int8 Quantized | int4 Quantized |
| ----- | -------- | --------- | -------------- | -------------- |
| 512   | 4071.7   | 3237.2    | 3338.2         | 3390.1         |
| 1024  | 6106.2   | 4372.4    | 4450.3         | 4359.1         |
| 2048  | 13971.8  | 12395.4   | 6736.8         | 6855.3         |
| 4096  | 46031.9  | 40621.4   | 22489.2        | 15971.1        |
| 8192  | 159163.1 | 156448.3  | 82325.7        | 49884.9        |
| 16384 | 691660.6 | 609235.0  | 310524.5       | 162752.0       |

Estimated BW (Max: 936.19 GB/s):

| Size  | Pytorch | My Kernel | int8 Quantized | int4 Quantized |
| ----- | ------- | --------- | -------------- | -------------- |
| 512   | 129.267 | 162.590   | 79.142         | 39.267         |
| 1024  | 344.117 | 480.571   | 236.540        | 121.214        |
| 2048  | 600.982 | 677.413   | 623.812        | 307.112        |
| 4096  | 729.295 | 826.432   | 746.741        | 526.263        |
| 8192  | 820.615 | 858.533   | 816.357        | 674.608        |
| 16384 | 776.300 | 881.759   | 865.513        | 826.690        |

Here are the parameters used for above results: 

| size  | blockDim.x | blockDim.y | gridDim.x | gridDim.y |
| ----- | ---------- | ---------- | --------- | --------- |
| 512   | 32         | 16         | 1         | 32        |
| 1024  | 32         | 32         | 1         | 32        |
| 2048  | 32         | 32         | 1         | 64        |
| 4096  | 256        | 1          | 1         | 4096      |
| 8192  | 256        | 4          | 1         | 2048      |
| 16384 | 512        | 2          | 1         | 8192      |

## Conclusion