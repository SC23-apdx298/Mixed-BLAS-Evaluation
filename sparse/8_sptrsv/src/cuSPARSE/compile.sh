DEFINE="-DCOMPUTE_TYPE=CUDA_R_64F -DVALUE_TYPE=double"
nvcc -I/usr/local/cuda/bin/../include spsv_csr_example.cu -o ./bin/sptrsv_csr_64 -lcudart -lcusparse $DEFINE -w -lnvidia-ml -lpthread -I../../include
echo "cusparse_csr_64"

DEFINE="-DCOMPUTE_TYPE=CUDA_R_32F -DVALUE_TYPE=float"
nvcc -I/usr/local/cuda/bin/../include spsv_csr_example.cu -o ./bin/sptrsv_csr_32 -lcudart -lcusparse $DEFINE -w -lnvidia-ml -lpthread -I../../include
echo "cusparse_csr_32"
