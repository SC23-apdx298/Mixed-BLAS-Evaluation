nvcc -O3 -w -arch=compute_61 -code=sm_86 -gencode=arch=compute_61,code=sm_86 -Xcompiler -fopenmp -Xcompiler -mfma main.cu -o ../bin/exe_spgemm_tile_64 -I/usr/local/cuda-11.1/include -L/usr/local/cuda-11.1/lib64  -lcudart  -lcusparse  -D VALUE_TYPE=double
echo "exe_spgemm_tile_64"

nvcc -O3 -w -arch=compute_61 -code=sm_86 -gencode=arch=compute_61,code=sm_86 -Xcompiler -fopenmp -Xcompiler -mfma main.cu -o ../bin/exe_spgemm_tile_32 -I/usr/local/cuda-11.1/include -L/usr/local/cuda-11.1/lib64  -lcudart  -lcusparse  -D VALUE_TYPE=float -D MAT_VAL_TYPE=float -D FLOAT
echo "exe_spgemm_tile_32"