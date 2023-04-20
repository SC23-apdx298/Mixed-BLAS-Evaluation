/*
   -- MAGMA (version 2.7.1) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date February 2023

   @author Azzam Haidar
   @author Tingxing Dong

   @generated from src/zgetf2_batched.cpp, normal z -> c, Wed Feb 22 15:19:43 2023
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

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
static magma_int_t
magma_cgetf2_batched_v1(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t **ipiv_array, magma_int_t *info_array,
    magma_int_t gbstep, magma_int_t batchCount,
    magma_queue_t queue)
{
    #define dAarray(i, j)  dA_array, i, j

    magma_int_t arginfo = 0;
    magmaFloatComplex c_neg_one = MAGMA_C_NEG_ONE;
    magmaFloatComplex c_one     = MAGMA_C_ONE;
    magma_int_t nb = BATF2_NB;

    magma_int_t min_mn = min(m, n);
    magma_int_t gbj, panelj, step, ib;

    for( panelj=0; panelj < min_mn; panelj += nb)
    {

        ib = min(nb, min_mn-panelj);

        for (step=0; step < ib; step++) {
            gbj = panelj+step;
            if ((m-panelj) > MAX_NTHREADS) {
                // find the max of the column
                arginfo = magma_icamax_batched(m-gbj, dA_array, ai+gbj, aj+gbj, ldda, 1, ipiv_array, ai+gbj, gbj, gbstep, info_array, batchCount, queue);
                if (arginfo != 0 ) return arginfo;
                // Apply the interchange to columns 1:N. swap the whole row
                arginfo = magma_cswap_batched(n, dA_array, ai, aj, ldda, gbj, ipiv_array, batchCount, queue);
                if (arginfo != 0 ) return arginfo;
                // Compute elements J+1:M of J-th column.
                if (gbj < m) {
                    arginfo = magma_cscal_cgeru_batched( m-gbj, ib-step, dA_array, ai+gbj, aj+gbj, ldda, info_array, gbj, gbstep, batchCount, queue );
                    if (arginfo != 0 ) return arginfo;
                }
            }
            else {
                arginfo = magma_ccomputecolumn_batched(m-panelj, panelj, step, dA_array, ai, aj, ldda, ipiv_array, info_array, gbstep, batchCount, queue);
                if (arginfo != 0 ) return arginfo;
                // Apply the interchange to columns 1:N. swap the whole row

                arginfo = magma_cswap_batched(n, dA_array, ai, aj, ldda, gbj, ipiv_array, batchCount, queue);
                if (arginfo != 0 ) return arginfo;
            }
        }


        if ( (n-panelj-ib) > 0) {
            // continue the update of the selected ib row column panelj+ib:n(TRSM)
            magma_cgetf2trsm_batched(ib, n-panelj-ib, dA_array, ai+panelj, ldda, batchCount, queue);
            // do the blocked DGER = DGEMM for the remaining panelj+ib:n columns
            magma_cgemm_batched_core( MagmaNoTrans, MagmaNoTrans, m-(panelj+ib), n-(panelj+ib), ib,
                                 c_neg_one, dAarray(ai+ib+panelj, aj+panelj   ), ldda,
                                            dAarray(ai+panelj   , aj+ib+panelj), ldda,
                                 c_one,     dAarray(ai+ib+panelj, aj+ib+panelj), ldda,
                                 batchCount, queue );
        }
    }

    return 0;

    #undef dAarray
}


static magma_int_t
magma_cgetf2_batched_v2(
    magma_int_t m, magma_int_t n, magma_int_t stop_nb,
    magmaFloatComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t **ipiv_array, magma_int_t** dpivinfo_array,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
#define dA_array(i,j) dA_array, i, j
#define ipiv_array(i) ipiv_array, i

    magma_int_t arginfo = 0;
    if(n <= stop_nb){
        arginfo = magma_cgetf2_fused_batched(m, n, dA_array(ai,aj), ldda, ipiv_array, info_array, batchCount, queue);
    }
    else{
        magma_int_t n1, n2;
        n1 = n/2;
        n2 = n - n1;

        // panel 1
        arginfo = magma_cgetf2_batched_v2(
                    m, n1, stop_nb,
                    dA_array(ai,aj), ldda,
                    ipiv_array, dpivinfo_array, info_array,
                    batchCount, queue);

        if(arginfo != 0) return arginfo;

        // swap right
        setup_pivinfo_batched(dpivinfo_array, ipiv_array(ai), m, n1, batchCount, queue);

        // fused update failed to launch, use classic update
        magma_claswp_rowparallel_batched(
                n2,
                dA_array(ai,aj+n1), ldda,
                dA_array(ai,aj+n1), ldda,
                0, n1, dpivinfo_array,
                batchCount, queue);

        // trsm
        magmablas_ctrsm_recursive_batched(
                MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                n1, n2, MAGMA_C_ONE,
                dA_array(ai,   aj), ldda,
                dA_array(ai,aj+n1), ldda,
                batchCount, queue );

        // gemm
        magma_cgemm_batched_core(
                MagmaNoTrans, MagmaNoTrans,
                m-n1, n2, n1,
                MAGMA_C_NEG_ONE, dA_array(ai+n1,    aj), ldda,
                                 dA_array(ai   , aj+n1), ldda,
                MAGMA_C_ONE,     dA_array(ai+n1, aj+n1), ldda,
                batchCount, queue );

        // panel 2
        magma_cgetf2_batched_v2(
                m-n1, n2, stop_nb,
                dA_array(ai+n1,aj+n1), ldda,
                ipiv_array, dpivinfo_array, info_array,
                batchCount, queue);

        // swap left
        setup_pivinfo_batched(dpivinfo_array, ipiv_array(ai+n1), m-n1, n2, batchCount, queue);
        adjust_ipiv_batched(ipiv_array(ai+n1), n2, n1, batchCount, queue);
        magma_claswp_rowparallel_batched(
                n1,
                dA_array(ai+n1,aj), ldda,
                dA_array(ai+n1,aj), ldda,
                n1, n, dpivinfo_array,
                batchCount, queue);
    }
    return arginfo;
#undef dA_array
#undef ipiv_array
}

extern "C" magma_int_t
magma_cgetf2_batched(
    magma_int_t m, magma_int_t n,
    magmaFloatComplex **dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
    magma_int_t **ipiv_array,
    magma_int_t** dpivinfo_array,
    magma_int_t *info_array,
    magma_int_t gbstep,
    magma_int_t batchCount,
    magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    if (m < 0) {
        arginfo = -1;
    } else if (n < 0 ) {
        arginfo = -2;
    } else if (ai < 0) {
        arginfo = -4;
    } else if (aj < 0 || aj != ai) {
        arginfo = -5;
    } else if (ldda < max(1,m)) {
        arginfo = -6;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }

    // first, test the fused panel
    arginfo = -1;
    for(magma_int_t inb = 32; inb >= 2; inb/=2 ) {
        arginfo = magma_cgetf2_batched_v2(m, n, inb, dA_array, ai, aj, ldda, ipiv_array, dpivinfo_array, info_array, batchCount, queue);
        if(arginfo == 0) break;
    }

    // negative arginfo means that fused panel did not launch
    if( arginfo != 0 ) {
        arginfo = magma_cgetf2_batched_v1(m, n, dA_array, ai, aj, ldda, ipiv_array, info_array, gbstep, batchCount, queue);
    }

    return arginfo;
}
