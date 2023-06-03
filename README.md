# FastGEMV

This repo provides a set of kernel functions that can compute GEMV (matrix-vector dot product) in high speed, both when the matrix is normal fp16 and when the matrix is in quantized int8/int4 with fp16 scale/zero point.

## Requirements

## Usage

```bash
$ make
$ ./gemv -s <matrix_size> -x <blockDim.x> -y <blockDim.y> -g <gridDim.x> -i <num_iterations> -b <bits_per_data>
# if using Nsight
$ nsys profile --stats=true --force-overwrite true -o report ./gemv -s <matrix_size> -x <blockDim.x> -y <blockDim.y> -g <gridDim.x> -i <num_iterations> -b <bits_per_data>
```

- `matrix_size`: Should be an integer power of 2, e.g.: 512, 1024, 2048, ... For example, when size = 512, the matrix is 512 * 512, and the vector is 512 * 1. 

- `blockDim.x` & `blockDim.y`: The block dimension when launching the kernel, both should be an integer power of 2. 

- `gridDim.x`: The grid dimension when launching the kernel. Only `gridDim.x` is needed, `gridDim.y` can be computed as `matrix_size / gridDim.x`.

- `num_iterations`: Number of iterations to do the test. If using Nsight to profile the runtime, it would be better to use a large iteration to get more precise values. 

- `bits_per_data`: If 16, the matrix is in fp16; if 8, the matrix is in int8, and if 4, the matrix is in int4. The vector, on the other hand, will always be in fp16. 

Other constraints:

- blockDim.y should be less than or equal to 64, based on the size of the shared memory used in the kernel.

- If using fp16 or int8, `size / (blockDim.x * gridDim.x)` should be greater than or equal to 8; if using int4, `size / (blockDim.x * gridDim.x)` should be greater than or equal to 16. 

## Workflow

When the `./gemv` program is running, it will first generate the matrix and vector data according to the size and bits provided by the user. All data will be generated using `curand`. Then the program will do `num_iterations` GEMV computations according to the blockDim and gridDim provided by the user. Finally, the program will check the result. If there is correctness error, it will print out the incorrect indexes as well as the values. If the test passes, no index will be printed out. 

The user can try multiple params of blockDim and gridDim to find out the best combinations for different settings and machines. 

## Optimization Strategy and Results

See [here](./method_and_result.md)