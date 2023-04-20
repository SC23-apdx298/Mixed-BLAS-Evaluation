# Mixed-BLAS-Evaluation

This is the code repository of SC '23 paper "Revealing the Practical Impact of Multi and Mixed Precision for Scientific Computing Kernels on GPU".

## System Configuration


* Relevant hardware details: NVIDIA RTX 4090.

* Operating systems: Ubuntu 22.04 LTS.

* Compilers: gcc v11.3.0, nvcc v12.0.140.

* NVIDIA driver version: 525.89.02.


## Libraries and Softwares Versions


* cuBLAS, 12.0.2.224,
<https://developer.nvidia.com/cublas>.

* cuFFT, 11.0.1.95,
<https://developer.nvidia.com/cufft>.

* cuSPARSE, 12.0.1.140,
<https://developer.nvidia.com/cusparse>.

* MAGMA, 2.7.1,
<https://icl.cs.utk.edu/magma>.

* Nsight Compute, 2022.4.1.6,
<https://developer.nvidia.com/nsight-compute>.


## Tools for Data Processing


* matplotlib, 3.5.2, <https://matplotlib.org>.

* seaborn, 0.11.2, <https://seaborn.pydata.org>.

* pandas, 1.4.4, <https://pandas.pydata.org>.

* numpy, 1.21.5, <https://numpy.org>.

## Steps to Reproduce the Results


* Each folder will have a script named compile.sh to compile all the precision of the kernel.

* And the executable file will be generated to bin which contains a test.sh to bench the performance results. (For sparse kernels, data-path in test.sh need to be adopted)

* To aquire the hardware statistics, analysis.sh will execute a massive profiling to obtain the NCU data used in paper.

* Jupyter notebooks plot all the appeared figure in manuscript.


Estimation of the execution time of all kernels is about 10 days.
FYI, I/O of sparse matrices wastes tons of time, by converting matrices into binary formats, which could be done through biio.h.



## Analysis Description

### Figure 4


* Filename: stream-ana.ipynb

* Raw data: stream-16.csv, stream-32.csv, stream-64.csv.


### Figure 5


* Filename: magma-conbine-gemm-getrf.ipynb

* Raw data: dgemm-magma-4090.csv,
                sgemm-magma-4090.csv,
                hgemm-magma-4090.csv,
                hgemm-magma-4090-notc.csv,
                dgetrf-magma-4090.csv,
                sgetrf-magma-4090.csv,
                hgetrf-magma-4090.csv.



### Figure 6


* Filename: ncu-gemm.py

* Raw data: ncu-hgemm.csv,
                ncu-sgemm.csv,
                ncu-dgemm.csv



### Figure 7


* Filename: cufft-4090-3d.ipynb

* Raw data: cufft-4090-3d-clean.csv



### Figure 8


* Filename: cufft-4090-3d-throughput.ipynb

* Raw data: cufft-4090-3d-clean.csv,
                ncu-cufft-3d-4090-t.csv




### Figure 9


* Filename: FDTD3d-4090.ipynb

* Raw data: ncu-fdtd3d-fp16-4090.csv,
                ncu-fdtd3d-fp32-4090.csv,
                ncu-fdtd3d-fp64-4090.csv,
                perf-fdtd3d-fp16-4090.csv,
                perf-fdtd3d-fp32-4090.csv,
                perf-fdtd3d-fp64-4090.csv.



### Figure 10


* Filename: spmv-data-plot-all.ipynb

* Raw data: spmv-cusparse-5-4090.csv,
                spmv-csr5-4090.csv,
                spmv-tile-4090.csv.



### Figure 11


* Filename: spmv-cu-4090-detail.ipynb

* Raw data: spmv-cusparse-1201-4090-70-dis.csv,
                ncu-spmv-cusparse12-4090-detail.csv (compressed).



### Figure 12


* Filename: spgemm-data-plot-all.ipynb

* Raw data: spgemm-cusparse-reuse-4090-15g.csv,
                spgemm-workest-cbd.csv,
                tile-ana-.csv,
                speck-4090.csv,
                tile-4090.csv.



### Figure 13


* Filename: sptrsv-data-plot-all.ipynb

* Raw data: spsv-cusparse12-4090.csv,
                level-info.csv,
                sptrsv-sync-free-4090.csv,
                sptrsv-rec-4090.csv.



### Figure 14


* Filename: sptrsv-cu-4090-scatter.ipynb

* Raw data: spsv-cusparse12-4090.csv (compressed),
                level-info.csv,
                ncu-sptrsv-cusparse-4090.csv.



