# BUSseq_gpu
This is the GPU (CUDA C) version of BUSseq (Batch Effects Correction With Unknown Subtypes for scRNA-seq Data). For the C++ parallel version, please see https://github.com/songfd2018/BUSseq-1.0.

# Optional dropout branch
Compared to the master branch, the code in the branch, BUSseq_gpu_optional_dropout, is updated and allows:
- Optional Dropout (Assumes that no dropout events will occur in a branch, used when the user has prior knowledge)
- Writing out preserved MCMC iterations (instead of storing them in RAM, this can reduce RAM usage)

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
    nvcc ./gpu_busseq_optional_dropout.cu -o ./gpu_busseq_optional_dropout --compiler-options -Wall
    # Actually running!
    ./gpu_busseq_optional_dropout -B 4 -N ./count_data/demo_dim.txt -G 3000 -K 5 -s 13579 \
        -c ./count_data/demo_count.txt -i 4000 -b 2000 -u 500 -w 1000 \
        -d ./count_data/demo_dropout.txt -o demo_output
```
    Inside demo_dim.txt is:
    300 300 200 200

    Inside demo_count.txt is:
    count data N x G, without row and column names.

    Inside demo_dropout.txt is:
    1 1 1 1
    
    The write-out-iterations option (-w) is activated with argument 1000.
    This means that for the 2000 preserved iterations, they will be stored temporarily every 1000 iterations.
    
### Meaning of the arguements
The meaning of the arguments that can be feed into the run are as follows:
```
Mandatory:
-B    Integer, the number of batches of the input data.
-N    String, the file name of a file containing information of the no. of samples in each batch.
      The file must contain, in order, no. of samples in batch1, no. of samples in batch2,..., etc.
      e.g. 300 300 200 200
-G    Integer, the number of genes in the input data.
-K    Integer, the number subtypes (cell types) existent in the data.
      Different values can be used to search for the best fit.
-c    String, the file name of the input count data.
      Please note that count data has to have the dimension N/samples (rows) x G/Genes (columns).
-i    Integer, the total number of iterations desired.

Optional:
-s    Integer, this option allows user to specify the initial seed instead of letting it randomly generated everytime.
-b    Integer, the number of burn-in iterations desired.
      Defaults to max(0.3*number_of_iterations, 500).
-u    Integer, the number of iterations for which p and \tau_0 will remain unchanged.
-r    Integer, the number of GBs of RAM available on your machine.
      The print-temporary-result-to-disk option will be activated if necessary.
-w    Integer, the number of iterations per which preserved iterations are written to disk. 
      An alternative to inputting RAM available.
      The print-temporary-result-to-disk option will certainly be activated.
-d    String, the file name of the dropout index, i.e. whether dropout is possible in a batch.
      1 is allow dropouts while 0 is not.
      For example, if there are four batches, and batches 2 and 4 does not accept dropout,
      then the file should contain the following:
      1 0 1 0
-p    N/A, no further argument required.
      This is a flag that prints all the preserved iterations on top of posterior samples.
      Default is not printing.
      Not available if the print-temporary-results-to-disk option is activated.
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
/gpu_busseq_optional_dropout -B 4 -N ./count_data/simulation_dim.txt -G 3000 -K 5 -s 13579 \
        -c ./count_data/simulation_count.txt -i 4000 -b 2000 -u 500 \
        -d ./count_data/simulation_dropout.txt -o simulation_output
```

## Contact
If you have any questions, please feel free to contact me via email (chan.ga.ming.angus@gmail.com) or leave an issue under this repository.
