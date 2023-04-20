#include <assert.h>
#include <cuda_fp16.h>
#include <cufft.h>
#include <cufftXt.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef double2 ftype;

#define BENCH_TIMES 1
#define WARMUP for (int warm_up = 0; warm_up < 0; warm_up++)
#define BENCH_TEST for (int test_round = 0; test_round < BENCH_TIMES; test_round++)

int main(int argc, char **argv)
{
    long long m = atoi(argv[1]);
    long long n = atoi(argv[2]);
    long long k = atoi(argv[3]);
    long long sig_idx[3];
    sig_idx[0] = 1 << m;
    sig_idx[1] = 1 << n;
    sig_idx[2] = 1 << k;
    long long sig_size = sig_idx[0] * sig_idx[1] * sig_idx[2];
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
    r = cufftXtMakePlanMany(plan, 3, sig_idx, NULL, 1, 1, CUDA_C_64F, NULL, 1,
                            1, CUDA_C_64F, 1, &ws, CUDA_C_64F);
    assert(r == CUFFT_SUCCESS);
    WARMUP
    {r = cufftXtExec(plan, d_idata, d_odata, CUFFT_FORWARD);} // warm-up
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
    et /= BENCH_TIMES;
    // printf("forward FFT time for %lld samples: %fms\n", sig_size, et);
    double gflops = (5.0 * sig_size * 1e-6 * log2(sig_size) / et);
    printf("3d,64,%lld,%f,%lf,%lld,%lld,%lld\n", sig_size, et, gflops, sig_idx[0], sig_idx[1], sig_idx[2]);

    return 0;
}