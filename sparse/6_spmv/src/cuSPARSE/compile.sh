# DEFINE="-DCOMPUTE_TYPE_AX=CUDA_R_8I -DVALUE_TYPE_AX=int8_t -DCOMPUTE_TYPE_Y=CUDA_R_32I -DVALUE_TYPE_Y=int32_t -DCOMPUTE_TYPE=CUDA_R_32I -DALPHA_TYPE=int"
# nvcc spmv_csr.cu -o ./bin/exe_spmv_cusparse_8_32_32I $DEFINE -lcudart -lcusparse -arch=sm_89 -w -I../../include -lnvidia-ml -lpthread
# echo "exe_spmv_cusparse_8_32_32I"

DEFINE="-DCOMPUTE_TYPE_AX=CUDA_R_8I -DVALUE_TYPE_AX=int8_t -DCOMPUTE_TYPE_Y=CUDA_R_32F -DVALUE_TYPE_Y=float -DCOMPUTE_TYPE=CUDA_R_32F -DALPHA_TYPE=float"
nvcc spmv_csr.cu -o ./bin/exe_spmv_cusparse_8_32_32 $DEFINE -lcudart -lcusparse -arch=sm_89 -w -I../../include -lnvidia-ml -lpthread
echo "exe_spmv_cusparse_8_32_32"

# DEFINE="-DCOMPUTE_TYPE_AX=CUDA_R_16F -DVALUE_TYPE_AX=__half -DCOMPUTE_TYPE_Y=CUDA_R_32F -DVALUE_TYPE_Y=float -DCOMPUTE_TYPE=CUDA_R_32F -DALPHA_TYPE=float"
# nvcc spmv_csr.cu -o ./bin/exe_spmv_cusparse_16_32_32 $DEFINE -lcudart -lcusparse -arch=sm_89 -w -I../../include -lnvidia-ml -lpthread
# echo "exe_spmv_cusparse_16_32_32"

DEFINE="-DCOMPUTE_TYPE_AX=CUDA_R_16BF -DVALUE_TYPE_AX=__nv_bfloat16 -DCOMPUTE_TYPE_Y=CUDA_R_32F -DVALUE_TYPE_Y=float -DCOMPUTE_TYPE=CUDA_R_32F -DALPHA_TYPE=float"
nvcc spmv_csr.cu -o ./bin/exe_spmv_cusparse_16BF_32_32 $DEFINE -lcudart -lcusparse -arch=sm_89 -w -I../../include -lnvidia-ml -lpthread
echo "exe_spmv_cusparse_16BF_32_32"

# DEFINE="-DCOMPUTE_TYPE_AX=CUDA_R_16F -DVALUE_TYPE_AX=__half -DCOMPUTE_TYPE_Y=CUDA_R_16F -DVALUE_TYPE_Y=__half -DCOMPUTE_TYPE=CUDA_R_32F -DALPHA_TYPE=float"
# nvcc spmv_csr.cu -o ./bin/exe_spmv_cusparse_16_16_32 $DEFINE -lcudart -lcusparse -arch=sm_89 -w -I../../include -lnvidia-ml -lpthread
# echo "exe_spmv_cusparse_16_16_32"

# DEFINE="-DCOMPUTE_TYPE_AX=CUDA_R_16BF -DVALUE_TYPE_AX=__nv_bfloat16 -DCOMPUTE_TYPE_Y=CUDA_R_16BF -DVALUE_TYPE_Y=__nv_bfloat16 -DCOMPUTE_TYPE=CUDA_R_32F -DALPHA_TYPE=float"
# nvcc spmv_csr.cu -o ./bin/exe_spmv_cusparse_16BF_16BF_32 $DEFINE -lcudart -lcusparse -arch=sm_89 -w -I../../include -lnvidia-ml -lpthread
# echo "exe_spmv_cusparse_16BF_16BF_32"

DEFINE="-DCOMPUTE_TYPE_AX=CUDA_R_32F -DVALUE_TYPE_AX=float -DCOMPUTE_TYPE_Y=CUDA_R_32F -DVALUE_TYPE_Y=float -DCOMPUTE_TYPE=CUDA_R_32F -DALPHA_TYPE=float"
nvcc spmv_csr.cu -o ./bin/exe_spmv_cusparse_32F $DEFINE -lcudart -lcusparse -arch=sm_89 -w -I../../include -lnvidia-ml -lpthread
echo "exe_spmv_cusparse_32F"

DEFINE="-DCOMPUTE_TYPE_AX=CUDA_R_64F -DVALUE_TYPE_AX=double -DCOMPUTE_TYPE_Y=CUDA_R_64F -DVALUE_TYPE_Y=double -DCOMPUTE_TYPE=CUDA_R_64F -DALPHA_TYPE=double"
nvcc spmv_csr.cu -o ./bin/exe_spmv_cusparse_64F $DEFINE -lcudart -lcusparse -arch=sm_89 -w -I../../include -lnvidia-ml -lpthread
echo "exe_spmv_cusparse_64F"
