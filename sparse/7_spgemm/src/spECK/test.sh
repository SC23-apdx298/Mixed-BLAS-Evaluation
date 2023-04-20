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
    for exe in `ls $1 | grep "exe"`
    do
        for data in `find $2 -name "*.mtx"`
        do
            if check $data; then
                echo -n $exe
                echo -n ","
                echo -n $data
                echo -n ","
                ./$1$exe $data config.ini
            fi
    	done
    done
}

exe_path="./bin/"
#data_path="/home/weifeng/MM/Engwirda/airfoil_2d/"
data_path="/home/weifeng/MM/"

test_all $exe_path $data_path

