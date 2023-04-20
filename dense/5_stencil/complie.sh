/usr/local/cuda/bin/nvcc -ccbin g++ -I./Common -Iinc -m64 --threads 0 --std=c++11 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o FDTD3d_fp16.o -c src/FDTD3d.cpp -DDTYPE=__half
/usr/local/cuda/bin/nvcc -ccbin g++ -I./Common -Iinc -m64 --threads 0 --std=c++11 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o FDTD3dGPU_fp16.o -c src/FDTD3dGPU.cu -DDTYPE=__half
/usr/local/cuda/bin/nvcc -ccbin g++ -I./Common -Iinc -m64 --threads 0 --std=c++11 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o FDTD3dReference_fp16.o -c src/FDTD3dReference.cpp -DDTYPE=__half
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o FDTD3d_fp16 FDTD3d_fp16.o FDTD3dGPU_fp16.o FDTD3dReference_fp16.o  -DDTYPE=__half
mkdir -p ./bin/x86_64/linux/release
cp FDTD3d_fp16 ./bin/x86_64/linux/release
/usr/local/cuda/bin/nvcc -ccbin g++ -I./Common -Iinc -m64 --threads 0 --std=c++11 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o FDTD3d_fp32.o -c src/FDTD3d_32.cpp -DDTYPE=float
/usr/local/cuda/bin/nvcc -ccbin g++ -I./Common -Iinc -m64 --threads 0 --std=c++11 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o FDTD3dGPU_fp32.o -c src/FDTD3dGPU_32.cu -DDTYPE=float
/usr/local/cuda/bin/nvcc -ccbin g++ -I./Common -Iinc -m64 --threads 0 --std=c++11 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o FDTD3dReference_fp32.o -c src/FDTD3dReference_32.cpp -DDTYPE=float
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o FDTD3d_fp32 FDTD3d_fp32.o FDTD3dGPU_fp32.o FDTD3dReference_fp32.o  -DDTYPE=float
mkdir -p ./bin/x86_64/linux/release
cp FDTD3d_fp32 ./bin/x86_64/linux/release
/usr/local/cuda/bin/nvcc -ccbin g++ -I./Common -Iinc -m64 --threads 0 --std=c++11 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o FDTD3d_fp64.o -c src/FDTD3d_64.cpp -DDTYPE=double
/usr/local/cuda/bin/nvcc -ccbin g++ -I./Common -Iinc -m64 --threads 0 --std=c++11 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o FDTD3dGPU_fp64.o -c src/FDTD3dGPU_64.cu -DDTYPE=double
/usr/local/cuda/bin/nvcc -ccbin g++ -I./Common -Iinc -m64 --threads 0 --std=c++11 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o FDTD3dReference_fp64.o -c src/FDTD3dReference_64.cpp -DDTYPE=double
/usr/local/cuda/bin/nvcc -ccbin g++ -m64 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o FDTD3d_fp64 FDTD3d_fp64.o FDTD3dGPU_fp64.o FDTD3dReference_fp64.o  -DDTYPE=double
mkdir -p ./bin/x86_64/linux/release
cp FDTD3d_fp64 ./bin/x86_64/linux/release
rm *.o