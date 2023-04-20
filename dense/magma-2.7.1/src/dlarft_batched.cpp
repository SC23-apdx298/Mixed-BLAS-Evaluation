/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @author Azzam Haidar
       @author Ahmad Abdelfattah

       @generated from src/zlarft_batched.cpp, normal z -> d, Wed Feb 22 15:19:46 2023
*/
#include "magma_internal.h"

#define  max_shared_bsiz 32

/******************************************************************************/
extern "C" void
magma_dlarft_sm32x32_batched(
        magma_int_t n, magma_int_t k,
        double **v_array, magma_int_t vi, magma_int_t vj, magma_int_t ldv,
        double **tau_array, magma_int_t taui,
        double **T_array, magma_int_t Ti, magma_int_t Tj, magma_int_t ldt,
        magma_int_t batchCount, magma_queue_t queue)
{
    if ( k <= 0) return;

    magma_dgemm_batched_core(
            MagmaConjTrans, MagmaNoTrans,
            k, k, n,
            MAGMA_D_ONE,
            v_array, vi, vj, ldv,
            v_array, vi, vj, ldv,
            MAGMA_D_ZERO,
            T_array, Ti, Tj, ldt,
            batchCount, queue );

    magmablas_dlaset_internal_batched(
            MagmaLower, k, k,
            MAGMA_D_ZERO, MAGMA_D_ZERO,
            T_array, Ti, Tj, ldt,
            batchCount, queue );

    // TRMV
    // T(1:i-1,i) := T(1:i-1,1:i-1) * W(1:i-1) i=[1:k]
    magmablas_dlarft_dtrmv_sm32x32_batched(
        k, k, tau_array, taui,
        T_array, Ti, Tj, ldt,
        T_array, Ti, Tj, ldt,
        batchCount, queue);
}


/******************************************************************************/
extern "C" magma_int_t
magma_dlarft_internal_batched(
        magma_int_t n, magma_int_t k, magma_int_t stair_T,
        double **v_array,   magma_int_t vi, magma_int_t vj, magma_int_t ldv,
        double **tau_array, magma_int_t taui,
        double **T_array,   magma_int_t Ti, magma_int_t Tj, magma_int_t ldt,
        double **work_array, magma_int_t lwork,
        magma_int_t batchCount, magma_queue_t queue)
{
    double c_one  = MAGMA_D_ONE;
    double c_zero = MAGMA_D_ZERO;

    if ( k <= 0) return 0;
    if ( stair_T > 0 && k <= stair_T) return 0;

    magma_int_t maxnb = max_shared_bsiz;

    magma_int_t info = 0;
    if (stair_T > 0 && stair_T > maxnb) {
        info = -3;
    }
    else if (lwork < k*ldt) {
        info = -10;
    }
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }

    magma_int_t DEBUG=0;
    magma_int_t nb = stair_T == 0 ? min(k,maxnb) : stair_T;

    magma_int_t i, j, prev_n, mycol, rows;

    double **dTstep_array  = NULL;

    magma_int_t Tstepi, Tstepj;
    if (k > nb) {
        dTstep_array = work_array;
        Tstepi = 0;
        Tstepj = 0;
    }
    else {
        dTstep_array = T_array;
        Tstepi = Ti;
        Tstepj = Tj;
    }

    magma_int_t ldtstep = ldt; //a enlever
    // stair_T = 0 meaning all T
    // stair_T > 0 meaning the triangular portion of T has been computed.
    //                    the value of stair_T is the nb of these triangulars

    magma_dgemm_batched_core( MagmaConjTrans, MagmaNoTrans,
                              k, k, n,
                              c_one,  v_array, vi, vj, ldv,
                                      v_array, vi, vj, ldv,
                              c_zero, dTstep_array, Tstepi, Tstepj, ldtstep,
                              batchCount, queue );

    magmablas_dlaset_internal_batched(
            MagmaLower, k, k, MAGMA_D_ZERO, MAGMA_D_ZERO,
            dTstep_array, 0, 0, ldtstep, batchCount, queue );

    //TRMV
    //T(1:i-1,i) := T(1:i-1,1:i-1) * W(1:i-1) i=[1:k]
    // TRMV is split over block of column of size nb
    // the update should be done from top to bottom so:
    // 1- a gemm using the previous computed columns
    //    of T to update rectangular upper protion above
    //    the triangle of my columns
    // 2- the columns need to be updated by a serial
    //    loop over of gemv over itself. since we limit the
    //    shared memory to nb, this nb column
    //    are split vertically by chunk of nb rows

    dim3 grid(1, 1, batchCount);

    for (j=0; j < k; j += nb) {
        prev_n =  j;
        mycol  =  min(nb, k-j);
        // note that myrow = prev_n + mycol;
        if (prev_n > 0 && mycol > 0) {
            if (DEBUG == 3) {
                printf("doing gemm on the rectangular portion of size %lld %lld of T(%lld,%lld)\n",
                        (long long) prev_n, (long long) mycol, (long long) 0, (long long) j );
            }

            magma_dgemm_batched_core( MagmaNoTrans, MagmaNoTrans,
                                 prev_n, mycol, prev_n,
                                 c_one,  T_array,            Ti,       Tj, ldt,
                                         dTstep_array, Tstepi+0, Tstepj+j, ldtstep,
                                 c_zero, T_array,            Ti,     Tj+j, ldt,
                                 batchCount, queue );

            // update my rectangular portion (prev_n,mycol) using sequence of gemv
            for (i=0; i < prev_n; i += nb) {
                rows = min(nb,prev_n-i);
                if (DEBUG == 3) {
                    printf("        doing recdtrmv on the rectangular portion of size %lld %lld of T(%lld,%lld)\n",
                            (long long) rows, (long long) mycol, (long long) i, (long long) j );
                }

                if (rows > 0 && mycol > 0) {
                    magmablas_dlarft_recdtrmv_sm32x32_batched(
                            rows, mycol,
                            tau_array,      taui+j,
                            T_array,          Ti+i,     Tj+j, ldt,
                            dTstep_array, Tstepi+j, Tstepj+j, ldtstep,
                            batchCount, queue);
                }
            }
        }

        // the upper rectangular protion is updated, now if needed update the triangular portion
        if (stair_T == 0) {
            if (DEBUG == 3) {
                printf("doing dtrmv on the triangular portion of size %lld %lld of T(%lld,%lld)\n",
                        (long long) mycol, (long long) mycol, (long long) j, (long long) j );
            }

            if (mycol > 0) {
                magmablas_dlarft_dtrmv_sm32x32_batched(
                        mycol, mycol,
                        tau_array,      taui+j,
                        dTstep_array, Tstepi+j, Tstepj+j, ldtstep,
                        T_array,          Ti+j,     Tj+j, ldt,
                        batchCount, queue);
            }
        }
    } // end of j

    return 0;
}

/******************************************************************************/
extern "C" magma_int_t
magma_dlarft_batched(magma_int_t n, magma_int_t k, magma_int_t stair_T,
                double **v_array, magma_int_t ldv,
                double **tau_array, double **T_array, magma_int_t ldt,
                double **work_array, magma_int_t lwork,
                magma_int_t batchCount, magma_queue_t queue)
{
    magma_dlarft_internal_batched(
        n, k, stair_T,
        v_array,   0, 0, ldv,
        tau_array, 0,
        T_array,   0, 0, ldt,
        work_array, lwork,
        batchCount, queue);

    return 0;
}
