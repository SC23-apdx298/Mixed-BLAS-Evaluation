ncu -o test -f --set full \
                --section MemoryWorkloadAnalysis_Tables \
                --page details \
                --csv \
                ./cufft_2d_16 22 4 \
                | grep -e 'L1/TEX Hit Rate' -e 'L2 Hit Rate' \

