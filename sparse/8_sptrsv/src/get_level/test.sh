# the output will be
# program, matirx, bytes of A & x, bytes of y, bytes of computation,
#              matrix columns, matrix rows, nnzs, execution time(ms)

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
    for exe in `ls $1 | grep "get_info"`
    do
        for data in `find $2 -name "*.cbd"`
        do
            if check $data; then
                # echo -n $data
                # echo -n ","
                ./$exe $data
            fi
    	done
    done
}

exe_path="./"
data_path="/ssget/MM/"

test_all $exe_path $data_path

