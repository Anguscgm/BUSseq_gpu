# BUSseq_gpu
This is the GPU (CUDA C) version of BUSseq (Batch Effects Correction With Unknown Subtypes for scRNA-seq Data). For the C++ parallel version, please see https://github.com/songfd2018/BUSseq-1.0.

## Getting started
The following are some simple instructions for the GPU version of BUSseq.

### Prerequisites
BUSseq_gpu requires a working GPU accessible under the current environment, as well as the compiler `nvcc` which compiles the CUDA C code.

### Installation
BUSseq_gpu does not require an installation.
All you have to do is to download the .cu file, or clone this repository to a desired location.

### Example
The code can be run on a gpu with the following lines.
Once compiled, the executable file can be run with different arguments without the need of compiling again.
```
# Compile the CUDA code
nvcc ./BUSseq_gpu.cu -o ./BUSseq_gpu --compiler-options -Wall
# Actually running!
./BUSseq_gpu -B 4 -N ./count_data/demo_dim.txt -G 3000 -K 5 -s 13579 \
    -c ./count_data/demo_count.txt -i 4000 -b 2000 -u 500 -o demo_output
```

### Meaning of the arguements
The meaning of the arguments that can be feed into the run are as follows:
```
-B    Integer, the number of batches of the input data.
-N    String, the file name of a file containing information of the no. of samples in each batch.
      The file must contain, in order, no. of samples in batch1, no. of samples in batch2,..., etc.
      e.g. 300 300 200 200
-G    Integer, the number of genes in the input data.
-K    Integer, the number subtypes (cell types) existent in the data.
      Different values can be used to search for the best fit.
-s    Integer, this option allows user to specify the initial seed instead of letting it randomly generated everytime.
-c    String, the file name of the input count data.
      Please note that count data has to have the dimension N/samples (rows) x G/Genes (columns).
-i    Integer, the total number of iterations desired.
-b    Integer, the number of burn-in iterations desired.
-u    Integer, the number of iterations for which p and \tau_0 will remain unchanged.
-p    N/A, no further argument required.
      This is a flag that prints all the preserved iterations on top of posterior samples.
      Default is not printing.
-o    String, the prefix of the output files.
```
Please note that:
- the count data file (-c) has to be samples/cells(rows) x genes(columns)
- the arguments are case-sensitive
- unfilled arguments will be filled by the code automatically, with either preset values or calculated from your other arguments

## Implementation
The following will demonstrate how to perform BUSseq_gpu on simulation and real datasets.
### Simulation
```
./BUSseq_gpu -B 4 -N ./count_data/simulation_dim.txt -G 3000 -K 5 -s 13579 \
    -c ./count_data/simulation_count.txt -i 4000 -b 2000 -u 500 -p -o simulation_output
```
