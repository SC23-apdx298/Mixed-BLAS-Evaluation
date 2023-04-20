#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// make sure that asserts are enabled
#undef NDEBUG
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#define A(i, j) &A[(i) + (j)*ld]
#define dA(i, j) &dA[(i) + (j)*ld]
#define dB(i, j) &dB[(i) + (j)*ld]
#define C2(i, j) &C2[(i) + (j)*ld]
#define LU(i, j) &LU[(i) + (j)*ld]

// #define PRINT_INFO

#ifdef PRINT_INFO
#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "", ##args)
#else
#define INFO_LOG(fmt, args...)
#endif

#ifndef PRINT_INFO
#define TEST_LOG(fmt, args...) fprintf(stdout, "16,getrf," fmt "", ##args)
#else
#define TEST_LOG(fmt, args...)
#endif

#define BENCH_TIMES 5
#define WARMUP for (int warm_up = 0; warm_up < 5; warm_up++)
#define BENCH_TEST for (int test_round = 0; test_round < BENCH_TIMES; test_round++)

int main(int argc, char **argv)
{
    TESTING_CHECK(magma_init());
    magma_print_environment();
    float *A;
    real_Double_t t1, t2;
    magmaFloat_ptr dA;
    magma_int_t m, n, maxn, lda, ldda, info;
    magma_int_t *piv;
    real_Double_t gflops, gpu_perf;

    magma_opts opts;
    opts.parse_opts(argc, argv);

    for (int itest = 0; itest < opts.ntest; ++itest)
    {
        m = opts.msize[itest];
        n = opts.nsize[itest];
        INFO_LOG("%%========================================================================\n");
        INFO_LOG("m=%lld, n=%lld\n", (long long)m, (long long)n);
        maxn = max(m, n);
        lda = max(1, maxn);
        ldda = magma_roundup(n, 32);
        gflops = FLOPS_SGETRF(m, n) / 1e9;
        TESTING_CHECK(magma_imalloc_cpu(&piv, maxn));
        TESTING_CHECK(magma_smalloc_pinned(&A, lda * n));
        magma_generate_matrix(opts, n, n, A, lda);

        TESTING_CHECK(magma_smalloc(&dA, ldda * n));

        magma_ssetmatrix(m, n, A, lda, dA, ldda, opts.queue);
        WARMUP
        {magma_hgetrf_native(m, n, dA, ldda, piv, &info);}

        t1 = magma_sync_wtime(opts.queue);
        BENCH_TEST
        {magma_hgetrf_native(m, n, dA, ldda, piv, &info);}
        t1 = magma_sync_wtime(opts.queue) - t1;
        t1 /= BENCH_TIMES;
        gpu_perf = gflops / t1;

        INFO_LOG("%lf,%lf\n", t1, gpu_perf);
        TEST_LOG("%d,%lf,%lf\n", m, gpu_perf, t1);

        magma_free_cpu(piv);
        magma_free_pinned(A);
        magma_free(dA);
    }

    return 0;
}