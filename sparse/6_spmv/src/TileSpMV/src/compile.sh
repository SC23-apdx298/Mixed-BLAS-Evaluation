/usr/local/cuda-11.1/bin/nvcc -O3 -w -arch=compute_61 -code=sm_86 -gencode=arch=compute_61,code=sm_86 main.cu -o ../bin/exe_spmv_tile_64 -I/usr/local/cuda-11.1/include -I/home/hemeng/NVIDIA_CUDA-11.1_Samples/common/inc -L/usr/local/cuda-11.1/lib64  -lcudart -Xcompiler -fopenmp -O3  -D MAT_VAL_TYPE=double -I../../../include/ -lnvidia-ml -lpthread
echo "spmv_tile_64"
/usr/local/cuda-11.1/bin/nvcc -O3 -w -arch=compute_61 -code=sm_86 -gencode=arch=compute_61,code=sm_86 main.cu -o ../bin/exe_spmv_tile_32 -I/usr/local/cuda-11.1/include -I/home/hemeng/NVIDIA_CUDA-11.1_Samples/common/inc -L/usr/local/cuda-11.1/lib64  -lcudart -Xcompiler -fopenmp -O3  -D MAT_VAL_TYPE=float -I../../../include/ -lnvidia-ml -lpthread
echo "spmv_tile_32"
