/*
   -- MAGMA (version 2.7.1) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date February 2023

   @author Ahmad Abdelfattah

   @generated from testing/testing_zgetrf_vbatched.cpp, normal z -> c, Wed Feb 22 15:20:32 2023
 */
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"  // internal header
#endif

#define PRECISION_c

// uncomment to use mkl's group batch interface
//#define USE_MKL_GETRF_BATCH

float get_LU_error(magma_int_t M, magma_int_t N,
                    magmaFloatComplex *A,  magma_int_t lda,
                    magmaFloatComplex *LU, magma_int_t *IPIV)
{
    magma_int_t min_mn = min(M, N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    magmaFloatComplex alpha = MAGMA_C_ONE;
    magmaFloatComplex beta  = MAGMA_C_ZERO;
    magmaFloatComplex *L, *U;
    float work[1], matnorm, residual;

    TESTING_CHECK( magma_cmalloc_cpu( &L, M*min_mn ));
    TESTING_CHECK( magma_cmalloc_cpu( &U, min_mn*N ));
    memset( L, 0, M*min_mn*sizeof(magmaFloatComplex) );
    memset( U, 0, min_mn*N*sizeof(magmaFloatComplex) );

    lapackf77_claswp( &N, A, &lda, &ione, &min_mn, IPIV, &ione);
    lapackf77_clacpy( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
    lapackf77_clacpy( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );

    for (j=0; j < min_mn; j++)
        L[j+j*M] = MAGMA_C_MAKE( 1., 0. );

    matnorm = lapackf77_clange("f", &M, &N, A, &lda, work);

    blasf77_cgemm("N", "N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*lda] = MAGMA_C_SUB( LU[i+j*lda], A[i+j*lda] );
        }
    }
    residual = lapackf77_clange("f", &M, &N, LU, &lda, work);

    magma_free_cpu( L );
    magma_free_cpu( U );

    return residual / (matnorm * N);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing cgetrf_batched
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops=0, magma_perf=0, magma_time=0, cpu_perf=0, cpu_time=0;
    real_Double_t   NbyM;
    float          error;
    magma_int_t     hA_size = 0, dA_size = 0, piv_size = 0;
    magma_int_t     seed = 0;
    magmaFloatComplex *hA, *hR, *hA_magma, *hTmp;
    magmaFloatComplex *dA;
    magmaFloatComplex **dA_array = NULL, **hA_array = NULL, **hR_array = NULL, **hdA_array = NULL;

    magma_int_t     **hipiv_array = NULL, **hdipiv_array = NULL, **dipiv_array = NULL, **dpivinfo_array;
    magma_int_t     *ipiv, *hinfo;
    magma_int_t     *dipiv, *dpivinfo, *dinfo;

    magma_int_t *h_M = NULL, *h_N = NULL, *h_lda  = NULL, *h_ldda = NULL, *h_min_mn = NULL;
    magma_int_t *d_M = NULL, *d_N = NULL, *d_ldda = NULL, *d_min_mn;
    magma_int_t iM, iN, max_M=0, max_N=0, max_minMN=0, max_MxN=0, info=0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t batchCount;
    int status = 0;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    float tol = opts.tolerance * lapackf77_slamch("E");

    batchCount = opts.batchcount;

    TESTING_CHECK( magma_imalloc_cpu(&h_M,      batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_N,      batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_lda,    batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_ldda,   batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&h_min_mn, batchCount) );
    TESTING_CHECK( magma_imalloc_cpu(&hinfo,    batchCount ));

    TESTING_CHECK( magma_imalloc(&d_M,      batchCount) );
    TESTING_CHECK( magma_imalloc(&d_N,      batchCount) );
    TESTING_CHECK( magma_imalloc(&d_ldda,   batchCount) );
    TESTING_CHECK( magma_imalloc(&d_min_mn, batchCount) );
    TESTING_CHECK( magma_imalloc(&dinfo,    batchCount ));

    TESTING_CHECK( magma_malloc_cpu((void**)&hA_array,  batchCount * sizeof(magmaFloatComplex*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&hR_array,  batchCount * sizeof(magmaFloatComplex*)) );
    TESTING_CHECK( magma_malloc_cpu((void**)&hdA_array, batchCount * sizeof(magmaFloatComplex*)) );
    TESTING_CHECK( magma_malloc(    (void**)&dA_array,  batchCount * sizeof(magmaFloatComplex*)) );

    TESTING_CHECK( magma_malloc_cpu((void**)&hipiv_array,    batchCount * sizeof(magma_int_t*) ));
    TESTING_CHECK( magma_malloc_cpu((void**)&hdipiv_array,   batchCount * sizeof(magma_int_t*) ));
    TESTING_CHECK( magma_malloc(    (void**)&dipiv_array,    batchCount * sizeof(magma_int_t*) ));
    TESTING_CHECK( magma_malloc(    (void**)&dpivinfo_array, batchCount * sizeof(magma_int_t*) ));

    printf("%%             max   max\n");
    printf("%% BatchCount   M     N    CPU Gflop/s (ms)   MAGMA Gflop/s (ms)   ||PA-LU||/(||A||*N)\n");
    printf("%%==========================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        seed = rand();
        for( int iter = 0; iter < opts.niter; ++iter ) {
            srand(seed);    // necessary to have the same sizes across different iterations

            iM = opts.msize[itest];
            iN = opts.nsize[itest];
            NbyM  = (real_Double_t)iN / (real_Double_t)iM;

            hA_size  = 0;
            dA_size  = 0;
            piv_size = 0;
            gflops   = 0;

            for(int s = 0; s < batchCount; s++) {
                h_M[s]      = 1 + (rand() % iM);
                h_N[s]      = max(1, (magma_int_t) round(NbyM * real_Double_t(h_M[s])) ); // try to keep the M/N ratio
                max_M       = (s == 0) ? h_M[s] : max(h_M[s], max_M);
                max_N       = (s == 0) ? h_N[s] : max(h_N[s], max_N);
                h_lda[s]    = h_M[s];
                h_ldda[s]   = h_lda[s]; //magma_roundup( h_M[s], opts.align );  // multiple of 32 by default
                h_min_mn[s] = min( h_M[s], h_N[s] );
                max_minMN   = (s == 0) ? h_min_mn[s] : max(h_min_mn[s], max_minMN);
                max_MxN     = (s == 0) ? h_M[s] * h_N[s] : max(h_M[s] * h_N[s], max_MxN);
                hA_size    += h_lda[s]  * h_N[s];
                dA_size    += h_ldda[s] * h_N[s];
                piv_size   += h_min_mn[s];
                gflops     += FLOPS_CGETRF( h_M[s], h_N[s] ) / 1e9;
            }

            TESTING_CHECK( magma_imalloc_cpu( &ipiv,     piv_size ));
            TESTING_CHECK( magma_cmalloc_cpu( &hA,       hA_size  ));
            TESTING_CHECK( magma_cmalloc_cpu( &hA_magma, hA_size  ));
            TESTING_CHECK( magma_cmalloc_pinned( &hR,    hA_size  ));

            TESTING_CHECK( magma_cmalloc( &dA,       dA_size ));
            TESTING_CHECK( magma_imalloc( &dipiv,    piv_size ));
            TESTING_CHECK( magma_imalloc( &dpivinfo, batchCount * max_M ));

            /* Initialize ptr arrays */
            hA_array [0]    = hA;
            hR_array [0]    = hR;
            hdA_array[0]    = dA;
            hipiv_array [0] = ipiv;
            hdipiv_array[0] = dipiv;
            for(int s = 1; s < batchCount; s++) {
                hA_array[s]     = hA_array[s-1]  + h_lda[s-1]  * h_N[s-1];
                hR_array[s]     = hR_array[s-1]  + h_lda[s-1]  * h_N[s-1];
                hdA_array[s]    = hdA_array[s-1] + h_ldda[s-1] * h_N[s-1];
                hipiv_array[s]  = hipiv_array[s-1]  + h_min_mn[s-1];
                hdipiv_array[s] = hdipiv_array[s-1] + h_min_mn[s-1];
            }
            magma_iset_pointer(dpivinfo_array, dpivinfo, 1, 0, 0, max_M, batchCount, opts.queue );

            /* Initialize hA and copy to hR */
            lapackf77_clarnv( &ione, ISEED, &hA_size, hA );
            memcpy(hR, hA, hA_size * sizeof(magmaFloatComplex));

            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            magma_setvector(batchCount, sizeof(magmaFloatComplex*), hdA_array, 1, dA_array, 1, opts.queue);
            magma_setvector(batchCount, sizeof(magma_int_t*), hdipiv_array, 1, dipiv_array, 1, opts.queue);
            magma_isetvector(batchCount, h_M,    1, d_M,    1, opts.queue);
            magma_isetvector(batchCount, h_N,    1, d_N,    1, opts.queue);
            magma_isetvector(batchCount, h_ldda, 1, d_ldda, 1, opts.queue);
            magma_isetvector(batchCount, h_min_mn, 1, d_min_mn, 1, opts.queue);

            for(int s = 0; s < batchCount; s++) {
                magma_csetmatrix( h_M[s], h_N[s],
                                  hR_array[s],  h_lda[s],
                                  hdA_array[s], h_ldda[s], opts.queue );
            }


            if(opts.version == 1) {
                // main API, with error checking and
                // workspace allocation
                magma_time = magma_sync_wtime( opts.queue );
                info = magma_cgetrf_vbatched(
                        d_M, d_N,
                        dA_array, d_ldda,
                        dipiv_array, dinfo,
                        batchCount, opts.queue);
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            }
            else {
                // advanced API, totally asynchronous,
                // but requires some setup
                magma_time = magma_sync_wtime( opts.queue );
                magma_int_t nb, recnb;
                magma_get_cgetrf_vbatched_nbparam(max_M, max_N, &nb, &recnb);
                info = magma_cgetrf_vbatched_max_nocheck(
                        d_M, d_N, d_min_mn,
                        max_M, max_N, max_minMN, max_MxN, nb, 32,
                        dA_array, d_ldda,
                        dipiv_array, dpivinfo_array, dinfo,
                        batchCount, opts.queue);
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            }
            magma_perf = gflops / magma_time;

            hTmp = hA_magma;
            for(int s = 0; s < batchCount; s++) {
                magma_cgetmatrix( h_M[s], h_N[s], hdA_array[s], h_ldda[s], hTmp, h_lda[s], opts.queue );
                hTmp += h_lda[s] * h_N[s];
            }

            // check info
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo, 1, hinfo, 1, opts.queue );
            for (int i=0; i < batchCount; i++) {
                if (hinfo[i] != 0 ) {
                    printf("magma_cgetrf_batched matrix %lld returned internal error %lld\n",
                            (long long) i, (long long) hinfo[i] );
                }
            }

            if (info != 0) {
                printf("magma_cgetrf_batched returned argument error %lld: %s.\n",
                        (long long) info, magma_strerror( info ));
            }

            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {

                #if defined(MAGMA_WITH_MKL) && defined(USE_MKL_GETRF_BATCH)
                magma_int_t *group_size = new magma_int_t[batchCount];
                for(magma_int_t ig = 0; ig < batchCount; ig++) group_size[ig] = 1;

                cpu_time = magma_wtime();
                lapackf77_cgetrf_batch(
                         h_M, h_N,
                         hA_array, h_lda,
                         hipiv_array, &batchCount, group_size, hinfo );
                cpu_time = magma_wtime() - cpu_time;

                delete[] group_size;

                #else
                cpu_time = magma_wtime();
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for (magma_int_t s=0; s < batchCount; s++) {
                    magma_int_t locinfo;
                    lapackf77_cgetrf(&h_M[s], &h_N[s], hA_array[s], &h_lda[s], hipiv_array[s], &locinfo);
                    if (locinfo != 0) {
                        printf("lapackf77_cgetrf matrix %lld returned error %lld: %s.\n",
                               (long long) s, (long long) locinfo, magma_strerror( locinfo ));
                    }
                }
                #if !defined (BATCHED_DISABLE_PARCPU) && defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;

                #endif

                cpu_perf = gflops / cpu_time;
            }

            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%10lld %5lld %5lld   %7.2f (%7.2f)    %7.2f (%7.2f) ",
                       (long long) batchCount, (long long) max_M, (long long) max_N,
                       cpu_perf, cpu_time*1000.,
                       magma_perf, magma_time*1000.  );
            }
            else {
                printf("%10lld %5lld %5lld     ---   (  ---  )    %7.2f (%7.2f) ",
                       (long long) batchCount, (long long) max_M, (long long) max_N,
                       magma_perf, magma_time*1000. );
            }

            if ( opts.check ) {
                magma_getvector( piv_size, sizeof(magma_int_t), dipiv, 1, ipiv, 1, opts.queue );
                hA_array[0] = hA_magma;
                for(int s = 1; s < batchCount; s++) {
                    hA_array[s] = hA_array[s-1] + h_lda[s-1] * h_N[s-1];
                }

                error = 0;
                #pragma omp parallel for reduction(max:error)
                for (int s=0; s < batchCount; s++) {
                    float err = 0;
                    for (int k=0; k < h_min_mn[s]; k++) {
                        if (hipiv_array[s][k] < 1 || hipiv_array[s][k] > h_M[s] ) {
                            printf("error for matrix %lld ipiv @ %lld = %lld (terminated on first detection)\n",
                                    (long long) s, (long long) k, (long long) hipiv_array[s][k] );
                            err = 1;
                            error = magma_max_nan( err, error );
                            break;
                        }
                    }

                    // if a pivot entry is invalid for a matrix, skip error checking
                    if (err == 1) {
                        continue;
                    }

                    err = get_LU_error( h_M[s], h_N[s], hR_array[s], h_lda[s], hA_array[s], hipiv_array[s]);
                    error = magma_max_nan( err, error );
                }

                bool okay = (error < tol);
                status += ! okay;
                printf("   %8.2e   %s\n", error, (okay ? "ok" : "failed") );
            }
            else {
                printf("     ---\n");
            }

            magma_free_cpu( hA );
            magma_free_cpu( hA_magma );
            magma_free_pinned( hR );
            magma_free_cpu( ipiv );

            magma_free( dA );
            magma_free( dipiv );
            magma_free( dpivinfo );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    magma_free( d_M );
    magma_free( d_N );
    magma_free( d_ldda );
    magma_free( d_min_mn );
    magma_free( dA_array );
    magma_free( dipiv_array );
    magma_free( dpivinfo_array );
    magma_free( dinfo );

    magma_free_cpu( h_M );
    magma_free_cpu( h_N );
    magma_free_cpu( h_lda );
    magma_free_cpu( h_ldda );
    magma_free_cpu( h_min_mn );
    magma_free_cpu( hA_array );
    magma_free_cpu( hR_array );
    magma_free_cpu( hdA_array );
    magma_free_cpu( hipiv_array );
    magma_free_cpu( hdipiv_array );
    magma_free_cpu( hinfo );

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
