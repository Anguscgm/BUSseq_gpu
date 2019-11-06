# BUSseq_gpu
This is the GPU (CUDA C) version of BUSseq (Batch Effects Correction With Unknown Subtypes for scRNA-seq Data). For the C++ parallel version, please see https://github.com/songfd2018/BUSseq-1.0.

## Getting started
The following are some simple instructions for using the GPU version of BUSseq.

### Installation
BUSseq_gpu does not require an installation.
All you have to do is to download the .cu file, or clone this repository to a desired location.

### Example
The code can be run on a gpu with the following lines.
```
# Compile the CUDA code
nvcc ./BUSseq_gpu.cu -o ./BUSseq_gpu --compiler-options -Wall
# Actually running!
./BUSseq_gpu -B 4 -N demo_dim.txt -G 3000 -K 5 -s 123 -c demo_count.txt -i 4000 -b 2000 -u 500 -p -o demo_output
```

### Meaning of the arguements
The meaning of the arguments that can be feed into the run are as follows:
