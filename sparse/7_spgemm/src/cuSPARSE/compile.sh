nvcc -I/usr/local/cuda/bin/../include spgemm_example.cu -o ./bin/exe_spgemm_cusparse_64 -lcudart -lcusparse -DCOMPUTE_TYPE=CUDA_R_64F -DVALUE_TYPE=double -w -I../../include -lnvidia-ml -lpthread
echo "spgemm_cusparse_64"

nvcc -I/usr/local/cuda/bin/../include spgemm_example.cu -o ./bin/exe_spgemm_cusparse_32 -lcudart -lcusparse -DCOMPUTE_TYPE=CUDA_R_32F -DVALUE_TYPE=float -w -I../../include -lnvidia-ml -lpthread
echo "spgemm_cusparse_32"

nvcc -I/usr/local/cuda/bin/../include spgemm_example.cu -o ./bin/exe_spgemm_cusparse_16 -lcudart -lcusparse -DCOMPUTE_TYPE=CUDA_R_16F -DVALUE_TYPE=__half -w -I../../include -lnvidia-ml -lpthread
echo "spgemm_cusparse_16"

nvcc -I/usr/local/cuda/bin/../include spgemm_example.cu -o ./bin/exe_spgemm_cusparse_bf16 -lcudart -lcusparse -DCOMPUTE_TYPE=CUDA_R_16BF -DVALUE_TYPE=__nv_bfloat16 -w -I../../include -lnvidia-ml -lpthread
echo "spgemm_cusparse_bf16"
