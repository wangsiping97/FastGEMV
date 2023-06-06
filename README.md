# FastGEMV

This repository provides a collection of kernel functions that enable high-speed computation of GEMV (matrix-vector dot product).

We have implemented and benchmarked the following scenarios:

- matrix: fp16, vector: fp16;
- matrix: int8 (quantized with fp16 scale/zero point), vector: fp16;
- matrix: int4 (quantized with fp16 scale/zero point), vector: fp16.

The matrix and vector sizes range from 512 to 16384.

On P100 GPUs, we achieved a maximum speedup of 2.7x compared to the PyTorch baseline. On 3090 Ti GPUs, we achieved a maximum speedup of 1.4x.

## Requirements

```bash
sudo apt install -y cuda-11-7 nsight-systems-2023.1.2 nsight-compute-2023.1.1
```

## Usage

### Running the Baseline (PyTorch)

Ensure that PyTorch is correctly installed.

```bash
# using Nsight
nsys profile --stats=true --force-overwrite true -o <report_name> python baseline.py -size <size>
```

For the baseline results, please refer to [here](./method_and_result.md).

### Running FastGEMV (this repository)

```bash
make
./gemv [-s <size> -x <blockDim.x> -y <blockDim.y> -i <num_iterations> -b <bits_per_data> -u <scale> -v <zero_point>]
# if using Nsight, the following command will generate detailed report of each function / kernel
nsys profile --stats=true --force-overwrite true -o <report_name> ./gemv \
    [-s <size> -x <blockDim.x> -y <blockDim.y> -i <num_iterations> -b <bits_per_data> -u <scale> -v <zero_point>]
```

- `size`: Should be a power of 2 integer, e.g.: 512, 1024, 2048, ... For example, when `size = 512`, the matrix is 512 *512, and the vector is 512* 1.

- `blockDim.x` and `blockDim.y`: The block dimension when launching the kernel. Both should be power of 2 integers.

- `num_iterations`: Number of iterations to perform the test. When profiling the runtime with Nsight, using a large iteration count provides more precise values.

- `bits_per_data`: Set to 16 for fp16 matrix, 8 for int8 matrix, and 4 for int4 matrix. The vector is always in fp16.

- `scale` and `zero_point`: Only applicable for the quantized version. Default values are 0.0625 for `scale` and 0.01 for `zero_point`.

Other constraints:

- `blockDim.x * blockDim.y` should be less than or equal to the maximum number of threads in a block (e.g., 1024).

- `blockDim.y` should be less than or equal to 64, based on the size of the shared memory used in the kernel.

- For fp16 or int8, `size / blockDim.x` should be greater than or equal to 8. For int4, `size / blockDim.x` should be greater than or equal to 16.

Example:

```bash
./gemv -s 16384 -x 32 -y 8 -i 10000
```

The above command runs a GEMV with a 16k *16k matrix and a 16k* 1 vector for 10000 iterations, using the following parameters:

- `blockDim` is set to (32, 8).
- `bits_per_data` is set to the default value of 16, indicating that both the matrix and vector are in fp16.
- No scale or zero_point values are required.

## Workflow

When running the `./gemv` program, it first generates the matrix and vector data based on the size and bits specified by the user. All data is generated using `curand`. Then the program performs GEMV computations for `num_iterations` based on the `blockDim` and `gridDim` generated from the user input. Finally, the program verifies the correctness of the result. If any errors are found, it prints the incorrect indexes and values. If the test passes, no indexes are printed.

Users can try different `blockDim` parameters to find the best combinations for different settings and hardware.

## Optimization Strategy and Results

Please refer to [here](./method_and_result.md) for more details.
