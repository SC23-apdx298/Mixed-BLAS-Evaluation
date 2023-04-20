echo "complie..."
./complie.sh

echo "start testing..."
mkdir data

echo "running perf..."
./testing/perf_fdtd3d_test/perf_fdtd3d_fp16.sh > ./data/perf_fdtd3d_fp16_4090.csv
./testing/perf_fdtd3d_test/perf_fdtd3d_fp32.sh > ./data/perf_fdtd3d_fp32_4090.csv
./testing/perf_fdtd3d_test/perf_fdtd3d_fp64.sh > ./data/perf_fdtd3d_fp64_4090.csv

echo "running ncu..."
./testing/ncu_fdtd3d_test/ncu_fdtd3d_16.sh > ./data/ncu_fdtd3d_fp16_4090.csv
./testing/ncu_fdtd3d_test/ncu_fdtd3d_32.sh > ./data/ncu_fdtd3d_fp32_4090.csv
./testing/ncu_fdtd3d_test/ncu_fdtd3d_64.sh > ./data/ncu_fdtd3d_fp64_4090.csv

echo "end"
