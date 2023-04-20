nvcc src/stream.cu -o ../bin/exe_stream_16 -Dreal=half -arch=sm_86 -I../include

nvcc src/stream.cu -o ../bin/exe_stream_32 -Dreal=float -arch=sm_86 -I../include

nvcc src/stream.cu -o ../bin/exe_stream_64 -Dreal=double -arch=sm_86 -I../include