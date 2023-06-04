# Method and Result

## Optimization Strategy

### fp16

### Quantized int8

### Quantized int4

## Result

### Bandwidth estimation method

If only 1 kernel is used (no mid result), we have: 

```
Estimated BW = (sizeof(mat) + sizeof(vec) + sizeof(result)) / T 
```

This is because we need to read each data of both the matrix and the vector once, and then write it to the result vector. More specifically, 

- when matrix is in fp16, BW = (2 * n^2 + 2 * 2n) / T (GB/s)

- when matrix is in int8, BW = (n^2 + 2 * 2n) / T (GB/s)

- when matrix is in int4, BW = (0.5 * n^2 + 2 * 2n) / T (GB/s)

where n is the matrix size and T is in ns. 

If 2 kernels are used (there is a mid result), we have: 

```
Estimated BW = (sizeof(mat) + sizeof(mid_result) * 2 + sizeof(vec) + sizeof(result)) / T
```

This is because we not only need to read each data and write to the result, we also need to write to `mid_result` in the first kernel and read it in the second kernel. The actual size of `mid_result` is determined by `gridDim.x`, which is number of blocks used per matrix row. 

### On P100

We only tested the runtime and estimated bandwidth for non-quantized fp16 matrices. Here is the result compared with pytorch:

Total GEMV kernel(s) average runtime (ns): 

| Size  | Pytorch  | My Kernel | Speedup |
|-------|----------|-----------|---------|
| 512   | 5891.6   | 4087.6    | 1.441   |
| 1024  | 9047.2   | 5559.0    | 1.627   |
| 2048  | 28231.4  | 18197.8   | 1.551   |
| 4096  | 100727.5 | 63754.9   | 1.580   |
| 8192  | 618377.7 | 231148.6  | 2.675   |
| 16384 | 1588132.0| 897847.3  | 1.769   |

Estimated BW (Max: 732 GB/s):

| Size  | Pytorch | My Kernel |
|-------|---------|-----------|
| 512   | 89.337  | 128.764   |
| 1024  | 232.254 | 377.990   |
| 2048  | 297.428 | 461.418   |
| 4096  | 333.284 | 526.561   |
| 8192  | 202.083 | 580.797   |
| 16384 | 320.796 | 598.026   |

Here are the parameters used for above results: 

| size  | 1 or 2 kernels | blockDim.x | blockDim.y | gridDim.x | gridDim.y |
|-------|----------------|------------|------------|-----------|-----------|
| 512   | 1              | 32         | 4          | 1         | 128       |
| 1024  | 1              | 32         | 4          | 1         | 256       |
| 2048  | 1              | 32         | 4          | 1         | 512       |
| 4096  | 2              | 32         | 4          | 8         | 1024      |
| 8192  | 2              | 32         | 2          | 16        | 4096      |
| 16384 | 2              | 32         | 8          | 4         | 2048      |

### On 3090

Here is the overall runtime and achieved bandwidth compared with pytorch. Note that pytorch doesn't support direct computation of quantized GEMV.

Total GEMV kernel(s) average runtime (ns): 

| Size | Pytorch | My Kernel | int8 Quantized | int4 Quantized |
|------|---------|-----------|----------------|----------------|
| 512  | 4071.7  | 3237.2    | 3338.2         | 3390.1         |
| 1024 | 6106.2  | 4372.4    | 4450.3         | 4359.1         |
| 2048 | 13971.8 | 12395.4   | 6736.8         | 6855.3         |
| 4096 | 46031.9 | 40621.4   | 22489.2        | 15971.1        |
| 8192 | 159163.1| 156448.3  | 82325.7        | 49884.9        |
| 16384| 691660.6| 609235.0  | 310524.5       | 162752.0       |

Estimated BW (Max: 936.19 GB/s):

| Size  | Pytorch | My Kernel | int8 Quantized | int4 Quantized |
|-------|---------|-----------|----------------|----------------|
| 512   | 129.267 | 162.590   | 79.142         | 39.267         |
| 1024  | 344.117 | 480.571   | 236.540        | 121.214        |
| 2048  | 600.982 | 677.413   | 623.812        | 307.112        |
| 4096  | 729.295 | 826.432   | 746.741        | 526.263        |
| 8192  | 820.615 | 858.533   | 816.357        | 674.608        |
| 16384 | 776.300 | 881.759   | 865.513        | 826.690        |

Here are the parameters used for above results: 

| size  | 1 or 2 kernels | blockDim.x | blockDim.y | gridDim.x | gridDim.y |
|-------|----------------|------------|------------|-----------|-----------|
| 512   | 1              | 32         | 16         | 1         | 32        |
| 1024  | 1              | 32         | 32         | 1         | 32        |
| 2048  | 1              | 32         | 32         | 1         | 64        |
| 4096  | 1              | 256        | 1          | 1         | 4096      |
| 8192  | 2              | 128        | 1          | 2         | 8192      |
| 16384 | 2              | 32         | 8          | 4         | 2048      |

## Future Work