#!/bin/bash -

for((i = 1160; i >= 96; i -= 8)) do
    echo -n ${i}
    echo -n ","
    ncu -o test -f --set full --section MemoryWorkloadAnalysis_Tables \
    --page details --csv \
    ./FDTD3d_fp16 -dimx=${i} -dimy=${i} -dimz=${i} \
    | tr -d '\n'
    echo ""
done
