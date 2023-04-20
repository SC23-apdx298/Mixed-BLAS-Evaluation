for((i=1000;i<=20000000;i+=1000));
do
    for((j=32;j<=1024;j*=2));
    do
        ./exe_stream_16 100 ${j} ${i}
        ./exe_stream_32 100 ${j} ${i}
        ./exe_stream_64 100 ${j} ${i}
    done
done
