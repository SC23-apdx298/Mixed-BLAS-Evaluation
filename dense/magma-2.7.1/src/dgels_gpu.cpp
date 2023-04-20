/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @generated from src/zgels_gpu.cpp, normal z -> d, Wed Feb 22 15:19:24 2023

*/
#include "magma_internal.h"

/***************************************************************************//**
    Purpose
    -------
    DGELS solves the overdetermined, least squares problem
           min || A*X - C ||
    using the QR factorization A.
    The underdetermined problem (m < n) is not currently handled.


    Arguments
    ---------
    @param[in]
    trans   magma_trans_t
      -     = MagmaNoTrans:   the linear system involves A.
            Only TRANS=MagmaNoTrans is currently handled.

    @param[in]
    m       INTEGER
            The number of rows of the matrix A. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A. M >= N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of columns of the matrix C. NRHS >= 0.

    @param[in,out]
    dA       DOUBLE PRECISION array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, A is overwritten by details of its QR
            factorization as returned by DGEQRF.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A, LDDA >= M.

    @param[in,out]
    dB      DOUBLE PRECISION array on the GPU, dimension (LDDB,NRHS)
            On entry, the M-by-NRHS matrix C.
            On exit, the N-by-NRHS solution matrix X.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB. LDDB >= M.

    @param[out]
    hwork   (workspace) DOUBLE PRECISION array, dimension MAX(1,LWORK).
            On exit, if INFO = 0, HWORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array HWORK,
            LWORK >= (M - N + NB)*(NRHS + NB) + NRHS*NB,
            where NB is the blocksize given by magma_get_dgeqrf_nb( M, N ).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the HWORK array, returns
            this value as the first entry of the HWORK array.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value

    @ingroup magma_gels
*******************************************************************************/
extern "C" magma_int_t
magma_dgels_gpu(
    magma_trans_t trans, magma_int_t m, magma_int_t n, magma_int_t nrhs,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dB, magma_int_t lddb,
    double *hwork, magma_int_t lwork,
    magma_int_t *info)
{
    magmaDouble_ptr dT;
    double *tau;
    magma_int_t min_mn;
    magma_int_t nb     = magma_get_dgeqrf_nb( m, n );
    magma_int_t lwkopt = (m - n + nb)*(nrhs + nb) + nrhs*nb;
    bool lquery = (lwork == -1);

    hwork[0] = magma_dmake_lwork( lwkopt );

    *info = 0;
    /* For now, N is the only case working */
    if ( trans != MagmaNoTrans )
        *info = -1;
    else if (m < 0)
        *info = -2;
    else if (n < 0 || m < n) /* LQ is not handle for now*/
        *info = -3;
    else if (nrhs < 0)
        *info = -4;
    else if (ldda < max(1,m))
        *info = -6;
    else if (lddb < max(1,m))
        *info = -8;
    else if (lwork < lwkopt && ! lquery)
        *info = -10;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery)
        return *info;

    min_mn = min(m,n);
    if (min_mn == 0) {
        hwork[0] = MAGMA_D_ONE;
        return *info;
    }

    /*
     * Allocate temporary buffers
     */
    magma_int_t ldtwork = ( 2*min_mn + magma_roundup( n, 32 ) )*nb;
    if (nb < nrhs)
        ldtwork = ( 2*min_mn + magma_roundup( n, 32 ) )*nrhs;
    if (MAGMA_SUCCESS != magma_dmalloc( &dT, ldtwork )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    
    magma_dmalloc_cpu( &tau, min_mn );
    if ( tau == NULL ) {
        magma_free( dT );
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    magma_dgeqrf_gpu( m, n, dA, ldda, tau, dT, info );

    if ( *info == 0 ) {
        magma_dgeqrs_gpu( m, n, nrhs,
                          dA, ldda, tau, dT,
                          dB, lddb, hwork, lwork, info );
    }
    
    magma_free( dT );
    magma_free_cpu( tau );
    return *info;
}
