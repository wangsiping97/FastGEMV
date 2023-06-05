# FastGEMV

This repo provides a set of kernel functions that can compute GEMV (matrix-vector dot product) in high speed.

We implemented and benchmarked the following cases:

- matrix: fp16, vector: fp16;
- matrix: int8 (quantized with fp16 scale/zero point), vector: fp16;
- matrix: int4 (quantized with fp16 scale/zero point), vector: fp16.

The size of the matrix and vector varies from 512 to 16384. 

On P100, we can have at most 2.7x speedup than pytorch baseline; on 3090 Ti, we can have at most 1.4x speedup.

## Requirements

```bash
$ sudo apt install -y cuda-11-7 nsight-systems-2023.1.2 nsight-compute-2023.1.1
```

## Usage

### How to run baseline (pytorch)

Make sure you have correctly installed pytorch.

```bash
# using Nsight
$ nsys profile --stats=true --force-overwrite true -o <report_name> python baseline.py -size <size>
```

For baseline result, see [here](./method_and_result.md).

### How to run fastGEMV (this repo)

```bash
$ make
$ ./gemv [-s <matrix_size> -x <blockDim.x> -y <blockDim.y> -i <num_iterations> -b <bits_per_data> -u <scale> -v <zero_point>]
# if using Nsight, the following command will generate detailed report of each function / kernel
$ nsys profile --stats=true --force-overwrite true -o <report_name> ./gemv \
    [-s <matrix_size> -x <blockDim.x> -y <blockDim.y> -i <num_iterations> -b <bits_per_data> -u <scale> -v <zero_point>]
```

- `matrix_size`: Should be an integer power of 2, e.g.: 512, 1024, 2048, ... For example, when size = 512, the matrix is 512 * 512, and the vector is 512 * 1.

- `blockDim.x` and `blockDim.y`: The block dimension when launching the kernel, both should be an integer power of 2. 

- `num_iterations`: Number of iterations to do the test. If using Nsight to profile the runtime, it would be better to use a large iteration to get more precise values. 

- `bits_per_data`: If set it to 16, the matrix is in fp16; if set to 8, the matrix is in int8, and if set to 4, the matrix is in int4. The vector, on the other hand, will always be in fp16. 

- `scale` and `zero_point`: Only available for quantized version. Default to 0.0625 for `scale` and 0.01 for `zero_point`.

Other constraints:

- `blockDim.x * blockDim.y` should be less than or equal to the maximum number of threads in a block (or 1024).

- `blockDim.y` should be less than or equal to 64, based on the size of the shared memory used in the kernel.

- If using fp16 or int8, `size / blockDim.x` should be greater than or equal to 8; if using int4, `size / blockDim.x` should be greater than or equal to 16. 

Example: 

```bash
$ ./gemv -s 16384 -x 32 -y 8 -i 10000
```

The above command tries to run a GEMV with 16k * 16k matrix and 16k * 1 vector for 10000 times, with the following parameters:

- `blockDim` is (32, 8);
- `bits_per_data` is defualt to 16, so both the matrix and vector are in fp16;
- no `scale` or `zero_point` is needed.

## Workflow

When the `./gemv` program is running, it will first generate the matrix and vector data according to the size and bits provided by the user. All data will be generated using `curand`. Then the program will do `num_iterations` GEMV computations according to the `blockDim` and `gridDim` as generated from the user input. Finally, the program will check the result. If there is correctness error, it will print out the incorrect indexes as well as the values. If the test passes, no index will be printed out. 

The user can try multiple params of `blockDim` to find out the best combinations for different settings and machines. 

## Optimization Strategy and Results

See [here](./method_and_result.md).
