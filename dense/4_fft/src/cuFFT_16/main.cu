#include <assert.h>
#include <cuda_fp16.h>
#include <cufft.h>
#include <cufftXt.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef half2 ftype;

#define BENCH_TIMES 1
#define WARMUP for (int warm_up = 0; warm_up < 0; warm_up++)
#define BENCH_TEST for (int test_round = 0; test_round < BENCH_TIMES; test_round++)

int main(int argc, char** argv) {
    long long n = atoi(argv[1]);
    long long sig_size = 1 << n;
    ftype *h_idata = (ftype *)malloc(sig_size * sizeof(ftype));
    ftype *d_idata;
    ftype *d_odata;
    cudaMalloc(&d_idata, sizeof(ftype) * sig_size);
    cudaMalloc(&d_odata, sizeof(ftype) * sig_size);
    cufftHandle plan;
    cufftResult r;
    r = cufftCreate(&plan);
    assert(r == CUFFT_SUCCESS);
    size_t ws = 0;
    r = cufftXtMakePlanMany(plan, 1, &sig_size, NULL, 1, 1, CUDA_C_16F, NULL, 1,
                            1, CUDA_C_16F, 1, &ws, CUDA_C_16F);
    assert(r == CUFFT_SUCCESS);
    WARMUP
    {r = cufftXtExec(plan, d_idata, d_odata, CUFFT_FORWARD);}  // warm-up
    assert(r == CUFFT_SUCCESS);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    BENCH_TEST
    {r = cufftXtExec(plan, d_idata, d_odata, CUFFT_FORWARD);}
    assert(r == CUFFT_SUCCESS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float et;
    cudaEventElapsedTime(&et, start, stop);
    et/=BENCH_TIMES;
    // printf("forward FFT time for %lld samples: %fms\n", sig_size, et);
    double gflops = (5.0 * sig_size * 1e-6 * log2(sig_size) / et);
    printf("1d,16,%lld,%f,%lf,%lld\n", sig_size, et, gflops, sig_size);

    return 0;
}