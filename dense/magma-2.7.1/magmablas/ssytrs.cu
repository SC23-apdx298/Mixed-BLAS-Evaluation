/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @generated from magmablas/zhetrs.cu, normal z -> s, Wed Feb 22 15:21:24 2023

       @author Stan Tomov
*/

#include "magma_internal.h"
#include "magma_templates.h"

#define REAL

#define dA(i_, j_) (dA + (i_) + (j_)*ldda)
#define dB(i_, j_) (dB + (i_) + (j_)*lddb)

#ifdef COMPLEX
void magmablas_slacgv( magma_int_t n, float *x, magma_int_t incx, magma_queue_t queue );
#endif

__global__ void
magma_sger_1(
    int n, int k, int kp, int nrhs, 
    float *A, float *B, int lddb)
{
    int tx  = threadIdx.x + 64 * blockIdx.x;

    if (k+1+tx < n)
        if (k!=kp && k+1+tx == kp)
            // if k <-> kp pivoting, B[k] holds the result for B[kp]
            B[k]  -= A[kp]*B[kp];
        else
            B[k+1+tx] -= A[k+1+tx]*B[kp];
}

__global__ void
magma_sswap_scal(
    int k, int kp, int nrhs, 
    float *A, float *B, int lddb)
{
    float tmp;
    if (k != kp){
        tmp   = B[k];
        B[k]  = B[kp];
        B[kp] = tmp;
    }
    B[k] *= MAGMA_S_DIV(MAGMA_S_ONE, A[k]);
}

__global__ void
magma_sger_2(
    int n, int k, int kp, int nrhs,
    float *A, int ldda, float *B, int lddb)
{
    int tx  = threadIdx.x + 64 * blockIdx.x;

    if (k+2+tx < n)
        if (k+1!=kp && k+2+tx == kp)
            // if k+1 <-> kp pivoting, B[k+1] holds the result for B[kp]
            B[k+1]  -= A[kp]*B[k] + A[kp+ldda]*B[kp];
        else
            B[k+2+tx] -= A[k+2+tx]*B[k] + A[k+2+tx+ldda]*B[kp];
}

__global__ void
magma_sswap_scal_inverseblock_lower(
    int k, int kp, int nrhs, 
    float *dA, int ldda, float *dB, int lddb)
{
    int tx  = threadIdx.x;

    float tmp;
    if (k+1 != kp){
        tmp         = *dB(k+1,tx);
        *dB(k+1,tx) = *dB(kp ,tx);
        *dB( kp,tx) = tmp;
    }

    float AKM1K = *dA(1,0);
    float AKM1  = MAGMA_S_DIV(*dA(0,0), MAGMA_S_CONJ( AKM1K ) );
    float AK    = MAGMA_S_DIV(*dA(1,1), AKM1K );
    float DENOM = AKM1*AK - MAGMA_S_ONE;

    float  BKM1 = MAGMA_S_DIV( *dB(k,tx),  MAGMA_S_CONJ(AKM1K));
    float  BK   = MAGMA_S_DIV( *dB(k+1,tx), AKM1K );

    *dB(k,tx) = MAGMA_S_DIV(  AK*BKM1-BK ,  DENOM );
    *dB(k+1,tx) = MAGMA_S_DIV( AKM1*BK-BKM1,  DENOM );
}

// This kernel scales the array B by 1/alpha.
// The kernel is called on one thread block with thread equal the 
// length of B, so that each thread scales just one element of B.
__global__ void
magmablas_sscal_inverse(
    float *alpha, 
    float *B, int ldb)
{
    int tx  = threadIdx.x;

    float scale = MAGMA_S_DIV(MAGMA_S_ONE, *alpha);
    B[tx*ldb] *= scale;
}

// Multiply array dB of size 2 by the inverse of the 2x2 diagonal block at dA.
// This is a batch operation where each thread is doing one multiplication.
__global__ void
magmablas_sscal_inverseblock_upper(
    const float *dA, int ldda, 
    float *dB, int lddb)
{
    int tx  = threadIdx.x;
    
    float AKM1K = *dA(0,1);
    float AKM1  = MAGMA_S_DIV(*dA(0,0), AKM1K);
    float AK    = MAGMA_S_DIV(*dA(1,1), MAGMA_S_CONJ( AKM1K ));
    float DENOM = AKM1*AK - MAGMA_S_ONE;

    float  BKM1 = MAGMA_S_DIV( *dB(0,tx), AKM1K);
    float  BK   = MAGMA_S_DIV( *dB(1,tx), MAGMA_S_CONJ(AKM1K) );

    *dB(0,tx) = MAGMA_S_DIV(  AK*BKM1-BK ,  DENOM );
    *dB(1,tx) = MAGMA_S_DIV( AKM1*BK-BKM1,  DENOM );
}

__global__ void
magmablas_sscal_inverseblock_lower(
    const float *dA, int ldda,
    float *dB, int lddb)
{
    int tx  = threadIdx.x;

    float AKM1K = *dA(1,0);
    float AKM1  = MAGMA_S_DIV(*dA(0,0), MAGMA_S_CONJ( AKM1K ) );
    float AK    = MAGMA_S_DIV(*dA(1,1), AKM1K );
    float DENOM = AKM1*AK - MAGMA_S_ONE;

    float  BKM1 = MAGMA_S_DIV( *dB(0,tx),  MAGMA_S_CONJ(AKM1K));
    float  BK   = MAGMA_S_DIV( *dB(1,tx), AKM1K );

    *dB(0,tx) = MAGMA_S_DIV(  AK*BKM1-BK ,  DENOM );
    *dB(1,tx) = MAGMA_S_DIV( AKM1*BK-BKM1,  DENOM );
}


/***************************************************************************//**
    Purpose
    -------
    SSYTRS solves a system of linear equations dA*dX = dB with a real
    symmetric matrix dA using the factorization dA = dU*dD*dU**H or
    dA = dL*dD*dL**H computed by SSYTRF_GPU.

    Arguments
    ---------

    @param[in]
    uplo    magma_uplo_t 
            Specifies whether the details of the factorization are stored
            as an upper or lower triangular matrix.
            = MagmaUpper:  Upper triangular, form is A = U*D*U**H;
            = MagmaLower:  Lower triangular, form is A = L*D*L**H.
    
    @param[in]
    n       INTEGER
            The order of the matrix dA.  N >= 0.
    
    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix dB.  NRHS >= 0.
    
    @param[in]
    dA      REAL array, dimension (LDA,N)
            The block diagonal matrix D and the multipliers used to
            obtain the factor U or L as computed by SSYTRF_GPU.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
    
    @param[in]
    ipiv    INTEGER array, dimension (N)
            Details of the interchanges and the block structure of D
            as determined by SSYTRF.
    
    @param[in,out]
    dB      REAL array, dimension (LDDB, NRHS)
            On entry, the right hand side matrix dB.
            On exit, the solution matrix dX.
    
    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  LDDB >= max(1,N).
    
    @param[out]
    info    INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_hetrs
*******************************************************************************/
extern "C" magma_int_t
magma_ssytrs_gpu(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
    float *dA, magma_int_t ldda,
    magma_int_t *ipiv,
    float *dB, magma_int_t lddb,
    magma_int_t *info, 
    magma_queue_t queue )
{
    /* Constants */
    const float c_one     = MAGMA_S_ONE;
    const float c_neg_one = MAGMA_S_NEG_ONE;

    /* Local variables */
    int k, kp;
    bool upper = (uplo == MagmaUpper);

    /* Test the input parameters. */
    *info = 0;
    if ( ! upper && uplo != MagmaLower ) {
        *info = -1;
    } else if ( n < 0 ) {
        *info = -2;
    } else if ( nrhs < 0 ) {
        *info = -3;
    } else if ( ldda < max( 1, n ) ) {
        *info = -5;
    } else if ( lddb < max( 1, n ) ) {
        *info = -8;
    }
    
    if ( *info != 0 ) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return *info;
    }
    
    if (upper) {
        /* Solve A*X = B, where A = U*D*U'.
           First solve U*D*X = B, overwriting B with X.
           K is the main loop index, decreasing from N to 1 in steps of
           1 or 2, depending on the size of the diagonal blocks.      */
        for(k = n-1; k > -1; k--) {
            if ( ipiv[k] > 0 ) {
                /* 1 x 1 diagonal block.
                   Interchange rows k and ipiv(k). */
                kp = ipiv[k]-1;
                if ( kp != k )
                    magma_sswap(nrhs, dB(k, 0), lddb, dB(kp, 0), lddb, queue);

                /* Multiply by inv(U(K)), where U(K) is the transformation
                   stored in column K of A. */
                magma_sger(k, nrhs, c_neg_one, dA(0,k), 1, dB(k,0), lddb, dB(0,0), lddb, queue);

                /* Multiply by the inverse of the diagonal block. */
                magmablas_sscal_inverse<<<1, nrhs, 0, queue->cuda_stream()>>>
                    (dA(k,k), dB(k,0), lddb);
            } 
            else {
                /* 2 x 2 diagonal block
                   Interchange rows K-1 and -IPIV(K). */
                kp = -ipiv[k]-1;
                if ( kp != k-1 )
                    magma_sswap(nrhs, dB(k-1,0), lddb, dB(kp,0), lddb, queue);

                /* Multiply by inv(U(K)), where U(K) is the transformation
                   stored in columns K-1 and K of A. */
                magma_sger(k-1, nrhs, c_neg_one, dA(0,k  ), 1, dB(k  ,0), lddb, dB, lddb, queue);
                magma_sger(k-1, nrhs, c_neg_one, dA(0,k-1), 1, dB(k-1,0), lddb, dB, lddb, queue);

                /* Multiply by the inverse of the diagonal block. */
                magmablas_sscal_inverseblock_upper<<<1, nrhs, 0, queue->cuda_stream()>>>
                    (dA(k-1,k-1), ldda, dB(k-1,0), lddb);

                /* reduce k once more for the 2 x 2 block */
                k--;
            }
        }

        /* Next solve U'*X = B, overwriting B with X.
           K is the main loop index, increasing from 1 to N in steps of
           1 or 2, depending on the size of the diagonal blocks.  */
        for(k=0; k<n; k++) {
            if ( ipiv[k] > 0) {
                /* 1 x 1 diagonal block
                   Multiply by inv(U'(K)), where U(K) is the transformation
                   stored in column K of A. */
                if (k > 0) {
                    #ifdef COMPLEX
                    magmablas_slacgv(nrhs, dB(k,0), lddb, queue);
                    #endif
                    magma_sgemv(MagmaConjTrans, k, nrhs, c_neg_one, dB, lddb,
                                dA(0, k), 1, c_one, dB(k, 0), lddb, queue);
                    #ifdef COMPLEX
                    magmablas_slacgv( nrhs, dB(k,0), lddb, queue);
                    #endif
                }

                /* Interchange rows K and IPIV(K). */
                kp = ipiv[k]-1;
                if( kp != k )
                    magma_sswap(nrhs, dB(k, 0), lddb, dB(kp, 0), lddb, queue);
            }
            else {
                /* 2 x 2 diagonal block
                   Multiply by inv(U'(K+1)), where U(K+1) is the transformation
                   stored in columns K and K+1 of A. */
                if (k > 0) {
                    #ifdef COMPLEX
                    magmablas_slacgv(nrhs, dB(k,0), lddb, queue);
                    #endif
                    magma_sgemv(MagmaConjTrans, k, nrhs, c_neg_one, dB, lddb,
                                dA(0, k), 1, c_one, dB(k,0), lddb, queue);
                    #ifdef COMPLEX
                    magmablas_slacgv(nrhs, dB(k,0), lddb, queue);
                    #endif

                    #ifdef COMPLEX
                    magmablas_slacgv(nrhs, dB(k+1,0), lddb, queue);
                    #endif
                    magma_sgemv(MagmaConjTrans, k, nrhs, c_neg_one, dB, lddb,
                                dA(0, k+1), 1, c_one, dB(k+1, 0), lddb, queue);
                    #ifdef COMPLEX
                    magmablas_slacgv(nrhs, dB(k+1,0), lddb, queue);
                    #endif
                }

                /* Interchange rows K and -IPIV(K). */
                kp = -ipiv[k]-1;
                if ( kp != k )
                    magma_sswap(nrhs, dB(k, 0), lddb, dB(kp, 0), lddb, queue);
            
                /* increase k one more for the 2 x 2 block */
                k++;
            }
        }    
    } else {
        /*  Solve A*X = B, where A = L*D*L'.
            First solve L*D*X = B, overwriting B with X.
    
            K is the main loop index, increasing from 1 to N in steps of
            1 or 2, depending on the size of the diagonal blocks. */
        for(k=0; k<n; k++) {
            if ( ipiv[k] > 0) {
                /* 1 x 1 diagonal block 
                   Interchange rows K and IPIV(K). */
                kp = ipiv[k]-1;
                if (0){
                if ( kp != k )
                    magma_sswap(nrhs, dB(k,0), lddb, dB(kp,0), lddb, queue);
                
                /* Multiply by inv(L(K)), where L(K) is the transformation
                   stored in column K of A.   */
                if (k < n-1)
                    magma_sger(n-k-1, nrhs, c_neg_one, dA(k+1, k), 1, dB(k,0), lddb,
                                dB(k+1, 0), lddb, queue);
                
                /* Multiply by the inverse of the diagonal block. */
                magmablas_sscal_inverse<<<1, nrhs, 0, queue->cuda_stream()>>>
                    (dA(k,k), dB(k,0), lddb);
                }     
                else {
                    magma_sger_1<<<magma_ceildiv(n-k-1,64), 64, 0, queue->cuda_stream()>>>
                        (n, k, kp, nrhs, dA(0,k), dB, lddb);
                    magma_sswap_scal<<<1, nrhs, 0, queue->cuda_stream()>>>
                        (k, kp, nrhs, dA(0,k), dB, lddb);
                }
            }
            else {
                /*  2 x 2 diagonal block
                    Interchange rows K+1 and -IPIV(K). */
                kp = -ipiv[k]-1;
                if (0) {
                if ( kp != k+1 )
                    magma_sswap(nrhs, dB(k+1,0), lddb, dB(kp,0), lddb, queue);

                /* Multiply by inv(L(K)), where L(K) is the transformation
                   stored in columns K and K+1 of A. */
                if ( k < n-2 ) {
                    magma_sger(n-k-2, nrhs, c_neg_one, dA(k+2,k), 1, dB(k,0), lddb,
                                dB(k+2,0), lddb, queue);
                    magma_sger(n-k-2, nrhs, c_neg_one, dA(k+2,k+1), 1, dB(k+1,0), lddb, 
                                dB(k+2,0), lddb, queue);
                }
            
                /* Multiply by the inverse of the diagonal block. */
                magmablas_sscal_inverseblock_lower<<<1, nrhs, 0, queue->cuda_stream()>>>
                    (dA(k,k), ldda, dB(k,0), lddb);
                }
                else {
                    magma_sger_2<<<magma_ceildiv(n-k-2,64), 64, 0, queue->cuda_stream()>>>
                        (n, k, kp, nrhs, dA(0,k), ldda, dB, lddb);
                    magma_sswap_scal_inverseblock_lower<<<1, nrhs, 0, queue->cuda_stream()>>>
                        (k, kp, nrhs, dA(k,k), ldda, dB(0, 0), lddb);
                }

                /* increase k one more for the 2 x 2 block */
                k++;
            }
        }
        
        /* Next solve L'*X = B, overwriting B with X.
           K is the main loop index, decreasing from N to 1 in steps of
           1 or 2, depending on the size of the diagonal blocks. */
        for(k = n-1; k > -1; k--) {
            if ( ipiv[k] > 0 ) {
                /* 1 x 1 diagonal block.
                   Multiply by inv(L'(K)), where L(K) is the transformation
                   stored in column K of A. */
                if (1){
                if ( k < n-1 ) {
                    #ifdef COMPLEX
                    magmablas_slacgv(nrhs, dB(k,0), lddb, queue);
                    #endif
                    magma_sgemv(MagmaConjTrans, n-k-1, nrhs, c_neg_one,
                                dB(k+1,0), lddb, dA(k+1,k), 1, c_one,
                                dB(k,0), lddb, queue);
                    #ifdef COMPLEX
                    magmablas_slacgv(nrhs, dB(k,0), lddb, queue);
                    #endif
                }

                /* Interchange rows K and IPIV(K). */
                kp = ipiv[k]-1;
                if ( kp!=k )
                    magma_sswap(nrhs, dB(k,0), lddb, dB(kp,0), lddb, queue);
                }
                else {
                }
            }
            else {
                /*  2 x 2 diagonal block
                    Multiply by inv(L'(K-1)), where L(K-1) is the transformation
                    stored in columns K-1 and K of A. */
                if ( k < n-1 ) {
                    #ifdef COMPLEX
                    magmablas_slacgv(nrhs, dB(k,0), lddb, queue);
                    #endif
                    magma_sgemv(MagmaConjTrans, n-k-1, nrhs, c_neg_one,
                                dB(k+1,0), lddb, dA(k+1,k), 1, c_one,
                                dB(k,0), lddb, queue);
                    #ifdef COMPLEX
                    magmablas_slacgv(nrhs, dB(k,0), lddb, queue);
                    #endif

                    #ifdef COMPLEX
                    magmablas_slacgv(nrhs, dB(k-1,0), lddb, queue);
                    #endif
                    magma_sgemv(MagmaConjTrans, n-k-1, nrhs, c_neg_one,
                                dB(k+1,0), lddb, dA(k+1,k-1), 1, c_one,
                                dB(k-1,0), lddb, queue);
                    #ifdef COMPLEX
                    magmablas_slacgv(nrhs, dB(k-1,0), lddb, queue);
                    #endif
                }

                /* Interchange rows K and -IPIV(K).*/
                kp = -ipiv[k]-1;
                if ( kp != k )
                    magma_sswap(nrhs, dB(k,0), lddb, dB(kp,0), lddb, queue);

                /* reduce k once more for the 2 x 2 block */
                k--;
            }
        }

    }
    return *info;
}
