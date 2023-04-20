#!/bin/bash -

for((i = 904; i >= 96; i -= 8)) do
    echo -n ${i}
    echo -n ","
    ./FDTD3d_fp64 -dimx=${i} -dimy=${i} -dimz=${i} \
    | grep -e "gpu_time" | tr -d '\n'
    echo ""
done
