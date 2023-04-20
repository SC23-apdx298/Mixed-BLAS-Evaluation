nvcc -O3 -w -m64 -Xptxas -dlcm=cg -arch=sm_86 main.cu -o ./bin/sptrsv_rec_64 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcusparse  -D VALUE_TYPE=double
echo "sptrsv_rec_64"

nvcc -O3 -w -m64 -Xptxas -dlcm=cg -arch=sm_86 main.cu -o ./bin/sptrsv_rec_32 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcusparse  -D VALUE_TYPE=float
echo "sptrsv_rec_32"
