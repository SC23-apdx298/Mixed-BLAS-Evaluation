#compilers
CC=nvcc

#GLOBAL_PARAMETERS
VALUE_TYPE_DOUBLE = double
VALUE_TYPE_FLOAT = float

#CUDA_PARAMETERS
#-Xcompiler -openmp 
NVCC_FLAGS = -O3 -w -m64 -Xptxas -dlcm=cg -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH = /usr/local/cuda

#includes
INCLUDES = -I$(CUDA_INSTALL_PATH)/include

#libs
#CLANG_LIBS = -stdlib=libstdc++ -lstdc++
#-lgomp 
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcusparse
LIBS = $(CUDA_LIBS)

#options
#OPTIONS = -std=c99

make:
	$(CC) $(NVCC_FLAGS) main.cu -o sptrsv_rec_64 $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE_DOUBLE)
	$(CC) $(NVCC_FLAGS) main.cu -o sptrsv_rec_32 $(INCLUDES) $(LIBS) $(OPTIONS) -D VALUE_TYPE=$(VALUE_TYPE_FLOAT)

test:
	./sptrsv-double -d 1 -rhs 1 -lv -1 -forward -mtx /home/hemeng/Documents/Data/matrices/webbase-1M.mtx