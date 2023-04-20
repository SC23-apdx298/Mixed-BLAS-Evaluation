check(){
    file_path=$1
    mat_name=${file_path##*/}
    mat_name=${mat_name%.*}
    parent_name=${file_path%/*}
    parent_name=${parent_name##*/}
    #echo $parent_name
    #echo $mat_name
    if [ "$parent_name" = "$mat_name" ]; then
        return 0 # 0 eq true
    else
        return 1
    fi
}

test_all(){
    for exe in `ls $1 | grep "exe"`
    do
        for data in `find $2 -name "*.cbd"`
        do
            if check $data; then
                echo -n $exe
                echo -n ","
                echo -n $data
                echo -n ","
                # echo ""
                ncu -o test -f --set full \
                --kernel-id ::csrmv_v3_kernel:9 \
                --section MemoryWorkloadAnalysis_Tables \
                --page details \
                --csv ./$1$exe \
                0 \
                $data \
                # | grep -e 'L1/TEX Hit Rate' -e 'L2 Hit Rate'
            fi
        done
    done
}

exe_path="./"
data_path="/ssget/MM"

test_all $exe_path $data_path

