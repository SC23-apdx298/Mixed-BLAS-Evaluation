all: fft_1d fft_2d fft_3d

fft_1d:
	make -C src/cuFFT_16
	make -C src/cuFFT_32
	make -C src/cuFFT_64

fft_2d:
	make -C src/cuFFT_2d_16
	make -C src/cuFFT_2d_32
	make -C src/cuFFT_2d_64

fft_3d:
	make -C src/cuFFT_3d_16
	make -C src/cuFFT_3d_32
	make -C src/cuFFT_3d_64
