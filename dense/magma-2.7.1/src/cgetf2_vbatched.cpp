/*
   -- MAGMA (version 2.7.1) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date February 2023

   @author Azzam Haidar
   @author Tingxing Dong

   @generated from src/zgetf2_vbatched.cpp, normal z -> c, Wed Feb 22 15:19:47 2023
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

static magma_int_t
magma_cgetf2_vbatched_v1(
    magma_int_t *m, magma_int_t *n, magma_int_t* minmn,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
    magmaFloatComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
    magma_int_t **ipiv_array, magma_int_t *info_array,
    magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)
{
#define dA_array(i,j) dA_array, (i), (j)
#define ipiv_array(i) ipiv_array, (i)

    magma_int_t arginfo = 0;

    magma_int_t j;
    magma_int_t nb = 8;

    for(j=0; j < max_minmn; j+=nb) {
        // panel (the swap is done on the entire width)
        magma_int_t ib = min(nb, max_minmn-j);
        for(magma_int_t jj = 0; jj < ib; jj++) {
            magma_int_t gbj = j+jj;
            // icamax
            magma_icamax_vbatched(max_m-gbj, m, n, dA_array(Ai+gbj, Aj+gbj), ldda, ipiv_array(Ai+gbj), info_array, gbj, gbstep, batchCount, queue);

            // cswap
            magma_cswap_vbatched(max_n, m, n, dA_array(Ai+gbj, Aj), ldda, ipiv_array, gbj, batchCount, queue);

            // scal+ger
            magma_cscal_cgeru_vbatched( max_m-gbj, ib-jj, m, n, dA_array(Ai+gbj, Aj+gbj), ldda, info_array, gbj, gbstep, batchCount, queue);
        }

        // trsm
        magmablas_ctrsm_vbatched_core(
            MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
            ib, max_n-j-ib, m, n, MAGMA_C_ONE,
            dA_array(Ai+j, Aj+j   ), ldda,
            dA_array(Ai+j, Aj+j+ib), ldda, batchCount, queue );

        // gemm
        magmablas_cgemm_vbatched_core(
            MagmaNoTrans, MagmaNoTrans,
            max_m-j-ib, max_n-j-ib, ib,
            m, n, minmn,
            MAGMA_C_NEG_ONE, dA_array, Ai+j+ib, Aj+j,    ldda,
                             dA_array, Ai+j,    Aj+j+ib, ldda,
            MAGMA_C_ONE,     dA_array, Ai+j+ib, Aj+j+ib, ldda,
            batchCount, queue );
    }

    return arginfo;

#undef dA_array
#undef ipiv_array
}



/***************************************************************************//**
    Purpose
    -------
    CGETF2 computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a batched version that factors batchCount M-by-N matrices in parallel.
    dA, ipiv, and info become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of each matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix A.  N >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ai      INTEGER
            Row offset for A.

    @param[in]
    aj      INTEGER
            Column offset for A.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    gbstep  INTEGER
            internal use.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    this is an internal routine that might have many assumption.


    @ingroup magma_getf2_batched
*******************************************************************************/
extern "C" magma_int_t
magma_cgetf2_vbatched(
    magma_int_t *m, magma_int_t *n, magma_int_t* minmn,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
    magmaFloatComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
    magma_int_t **ipiv_array, magma_int_t *info_array,
    magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;

    // first, test the fused panel
    arginfo = magma_cgetf2_fused_vbatched(
                max_m, max_n, max_minmn, max_mxn,
                m, n,
                dA_array, Ai, Aj, ldda,
                ipiv_array, Ai, info_array,
                batchCount, queue);
    if(arginfo == 0) return arginfo;

    magma_cgetf2_vbatched_v1(
        m, n,
        minmn, max_m, max_n, max_minmn, max_mxn,
        dA_array, Ai, Aj, ldda,
        ipiv_array, info_array, gbstep,
        batchCount, queue);

    return 0;
}

