import torch
import argparse


# function to generate random tensors
def gen(size):
    return (
        torch.rand((size, size), dtype=torch.float16).cuda(),
        torch.rand((size, 1), dtype=torch.float16).cuda(),
    )


# Create the parser and add the "-size" argument
parser = argparse.ArgumentParser(description="Process size argument.")
parser.add_argument("-size", type=int, required=True, help="Size of the tensor for matmul")

# Parse the arguments
args = parser.parse_args()

# Retrieve the size argument
size = args.size

# Generate the tensors and perform the matmul operation
a, b = gen(size)
for i in range(10000):
    c = a.matmul(b)
