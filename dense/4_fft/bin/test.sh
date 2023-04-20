#!/bin/bash

for((i=1;i<=28;i++));
do
    ./cufft_16 $i
    ./cufft_32 $i
    ./cufft_64 $i
done

for((i=1;i<=14;i++));
do
    ./cufft_2d_16 $i
    ./cufft_2d_32 $i
    ./cufft_2d_64 $i
done

for((i=1;i<=10;i++));
do
    ./cufft_3d_16 $i
    ./cufft_3d_32 $i
    ./cufft_3d_64 $i
done
