/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @precisions normal z -> c d s
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

extern "C" magma_int_t
magma_zpotrf_expert(magma_uplo_t uplo, magma_int_t n,
                    magmaDoubleComplex  *A, magma_int_t lda,
                    magmaDoubleComplex *dA, magma_int_t ldda,
                    magma_int_t *info, magma_queue_t *queues);

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zpotrf
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    // constants
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magma_int_t ione = 1;
    
    // locals
    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf, cpu_time;
    magmaDoubleComplex *h_A, *h_R;
    magma_int_t N, n2, lda, info;
    double      Anorm, error, work[1], *sigma;
    int status = 0;

    magma_opts opts;
    opts.matrix = "rand_dominant";  // default
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%% ngpu = %lld, uplo = %s\n", (long long) opts.ngpu, lapack_uplo_const(opts.uplo) );
    printf("%%   N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   ||R_magma - R_lapack||_F / ||R_lapack||_F\n");
    printf("%%=======================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N     = opts.nsize[itest];
            lda   = N;
            n2    = lda*N;
            gflops = FLOPS_ZPOTRF( N ) / 1e9;
            
            TESTING_CHECK( magma_zmalloc_cpu( &h_A, n2 ));
            TESTING_CHECK( magma_dmalloc_cpu( &sigma, N ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_R, n2 ));
            
            /* Initialize the matrix */
            magma_generate_matrix( opts, N, N, h_A, lda, sigma );
            lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            if (opts.verbose) {
                printf( "A = " ); magma_zprint( N, N, h_A, lda );
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            if (opts.version == 1) {
                gpu_time = magma_wtime();
                magma_zpotrf( opts.uplo, N, h_R, lda, &info );
                gpu_time = magma_wtime() - gpu_time;
            } 
            else {
                magmaDoubleComplex_ptr dA = NULL;
                magma_int_t ldda = magma_roundup( N, 32 );
                if (MAGMA_SUCCESS != magma_zmalloc( &dA, N*ldda ))
                    printf("Can not allocate GPU memory\n");
                
                magma_queue_t queues[2] = { NULL, NULL };
                magma_device_t cdev;
                magma_getdevice( &cdev );
                magma_queue_create( cdev, &queues[0] );
                magma_queue_create( cdev, &queues[1] );
                
                gpu_time = magma_wtime();
                magma_zpotrf_expert(opts.uplo, N, h_R, lda, dA, ldda, &info, queues );
                gpu_time = magma_wtime() - gpu_time;

                magma_queue_destroy( queues[0] );
                magma_queue_destroy( queues[1] );

                magma_free( dA );
            }
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zpotrf returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                cpu_time = magma_wtime();
                lapackf77_zpotrf( lapack_uplo_const(opts.uplo), &N, h_A, &lda, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zpotrf returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                /* =====================================================================
                   Check the result compared to LAPACK
                   =================================================================== */
                blasf77_zaxpy(&n2, &c_neg_one, h_A, &ione, h_R, &ione);
                #ifndef MAGMA_HAVE_HIP
                Anorm = lapackf77_zlange("f", &N, &N, h_A, &lda, work);
                error = lapackf77_zlange("f", &N, &N, h_R, &lda, work) / Anorm;
                #else
                // TODO: use zlange when the herk/syrk implementations are standardized. 
                // For HIP, the current herk/syrk routines overwrite the entire diagonal
                // blocks of the matrix, so using zlange causes the error check to fail
                Anorm = safe_lapackf77_zlanhe( "f", lapack_uplo_const(opts.uplo), &N, h_A, &lda, work );
                error = safe_lapackf77_zlanhe( "f", lapack_uplo_const(opts.uplo), &N, h_R, &lda, work ) / Anorm;
                #endif
                
                printf("%5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                       (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time,
                       error, (error < tol ? "ok" : "failed") );
                status += ! (error < tol);
            }
            else {
                printf("%5lld     ---   (  ---  )   %7.2f (%7.2f)     ---  \n",
                       (long long) N, gpu_perf, gpu_time );
            }
            magma_free_cpu( h_A );
            magma_free_cpu( sigma );
            magma_free_pinned( h_R );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
