/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @generated from include/magma_zvbatched.h, normal z -> d, Wed Feb 22 15:21:05 2023
*/

#ifndef MAGMA_DVBATCHED_H
#define MAGMA_DVBATCHED_H

#include "magma_types.h"

#define MAGMA_REAL

#ifdef __cplusplus
extern "C" {
#endif

  /*
   *  control and tuning
   */
void magma_get_dgetrf_vbatched_nbparam(magma_int_t max_m, magma_int_t max_n, magma_int_t *nb, magma_int_t *recnb);


  /*
   *  LAPACK vbatched routines
   */

magma_int_t
magma_dgetf2_fused_vbatched(
    magma_int_t max_M, magma_int_t max_N,
    magma_int_t max_minMN, magma_int_t max_MxN,
    magma_int_t* M, magma_int_t* N,
    double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t **dipiv_array, magma_int_t ipiv_i,
    magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue);

magma_int_t
magma_dgetf2_fused_sm_vbatched(
    magma_int_t max_M, magma_int_t max_N, magma_int_t max_minMN, magma_int_t max_MxN,
    magma_int_t* m, magma_int_t* n,
    double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t** dipiv_array, magma_int_t ipiv_i,
    magma_int_t* info_array, magma_int_t gbstep,
    magma_int_t nthreads, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_dgetrf_vbatched(
        magma_int_t* m, magma_int_t* n,
        double **dA_array, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_dgetrf_vbatched_max_nocheck(
        magma_int_t* m, magma_int_t* n, magma_int_t* minmn,
        magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
        magma_int_t nb, magma_int_t recnb,
        double **dA_array, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t** pivinfo_array,
        magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_dgetrf_vbatched_max_nocheck_work(
        magma_int_t* m, magma_int_t* n,
        magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
        double **dA_array, magma_int_t *ldda,
        magma_int_t **dipiv_array, magma_int_t *info_array,
        void* work, magma_int_t* lwork,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_idamax_vbatched(
        magma_int_t length, magma_int_t *M, magma_int_t *N,
        double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t** ipiv_array, magma_int_t ipiv_i,
        magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_dswap_vbatched(
        magma_int_t max_n, magma_int_t *M, magma_int_t *N,
        double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t** ipiv_array, magma_int_t piv_adjustment,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t magma_dscal_dger_vbatched(
    magma_int_t max_M, magma_int_t max_N,
    magma_int_t *M, magma_int_t *N,
    double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
    magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_dgetf2_vbatched(
    magma_int_t *m, magma_int_t *n, magma_int_t *minmn,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
    double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
    magma_int_t **ipiv_array, magma_int_t *info_array,
    magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_dgetrf_recpanel_vbatched(
    magma_int_t* m, magma_int_t* n, magma_int_t* minmn,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn,
    magma_int_t max_mxn, magma_int_t min_recpnb,
    double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t** dipiv_array, magma_int_t dipiv_i, magma_int_t** dpivinfo_array,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount,  magma_queue_t queue);

void
magma_dlaswp_left_rowserial_vbatched(
        magma_int_t n,
        magma_int_t *M, magma_int_t *N, double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t ipiv_offset,
        magma_int_t k1, magma_int_t k2,
        magma_int_t batchCount, magma_queue_t queue);

void
magma_dlaswp_right_rowserial_vbatched(
        magma_int_t n,
        magma_int_t *M, magma_int_t *N, double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t ipiv_offset,
        magma_int_t k1, magma_int_t k2,
        magma_int_t batchCount, magma_queue_t queue);

void
magma_dlaswp_left_rowparallel_vbatched(
        magma_int_t n,
        magma_int_t* M, magma_int_t* N,
        double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t k1, magma_int_t k2,
        magma_int_t **pivinfo_array, magma_int_t pivinfo_i,
        magma_int_t batchCount, magma_queue_t queue);

void
magma_dlaswp_right_rowparallel_vbatched(
        magma_int_t n,
        magma_int_t* M, magma_int_t* N,
        double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t k1, magma_int_t k2,
        magma_int_t **pivinfo_array, magma_int_t pivinfo_i,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_dpotrf_lpout_vbatched(
    magma_uplo_t uplo, magma_int_t *n, magma_int_t max_n,
    double **dA_array, magma_int_t *lda, magma_int_t gbstep,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_dpotf2_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n,
    double **dA_array, magma_int_t* lda,
    double **dA_displ,
    double **dW_displ,
    double **dB_displ,
    double **dC_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_dpotrf_panel_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n,
    magma_int_t *ibvec, magma_int_t nb,
    double** dA_array,    magma_int_t* ldda,
    double** dX_array,    magma_int_t* dX_length,
    double** dinvA_array, magma_int_t* dinvA_length,
    double** dW0_displ, double** dW1_displ,
    double** dW2_displ, double** dW3_displ,
    double** dW4_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_dpotrf_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_int_t *n,
    double **dA_array, magma_int_t *ldda,
    magma_int_t *info_array,  magma_int_t batchCount,
    magma_int_t max_n, magma_queue_t queue);

magma_int_t
magma_dpotrf_vbatched(
    magma_uplo_t uplo, magma_int_t *n,
    double **dA_array, magma_int_t *ldda,
    magma_int_t *info_array,  magma_int_t batchCount,
    magma_queue_t queue);
  /*
   *  BLAS vbatched routines
   */
/* Level 3 */
void
magmablas_dgemm_vbatched_core(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    double beta,
    double              ** dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dgemm_vbatched_max_nocheck(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dgemm_vbatched_max(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dgemm_vbatched_nocheck(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dgemm_vbatched(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsyrk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc,
    magma_int_t max_n, magma_int_t max_k,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsyrk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc,
    magma_int_t max_n, magma_int_t max_k,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsyrk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double beta,
    double **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_dsyrk_vbatched_max(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        double alpha,
        double const * const * dA_array, magma_int_t* ldda,
        double beta,
        double **dC_array, magma_int_t* lddc,
        magma_int_t batchCount,
        magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_dsyrk_vbatched_nocheck(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        double alpha,
        double const * const * dA_array, magma_int_t* ldda,
        double beta,
        double **dC_array, magma_int_t* lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsyrk_vbatched(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        double alpha,
        double const * const * dA_array, magma_int_t* ldda,
        double beta,
        double **dC_array, magma_int_t* lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsyrk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double beta,
    double **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_dsyrk_vbatched_max(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        double alpha,
        double const * const * dA_array, magma_int_t* ldda,
        double beta,
        double **dC_array, magma_int_t* lddc,
        magma_int_t batchCount,
        magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_dsyrk_vbatched_nocheck(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        double alpha,
        double const * const * dA_array, magma_int_t* ldda,
        double beta,
        double **dC_array, magma_int_t* lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsyrk_vbatched(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        double alpha,
        double const * const * dA_array, magma_int_t* ldda,
        double beta,
        double **dC_array, magma_int_t* lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsyr2k_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta, double **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_dsyr2k_vbatched_max(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta, double **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_dsyr2k_vbatched_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta, double **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsyr2k_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta, double **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsyr2k_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_dsyr2k_vbatched_max(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_dsyr2k_vbatched_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsyr2k_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    double alpha,
    double const * const * dA_array, magma_int_t* ldda,
    double const * const * dB_array, magma_int_t* lddb,
    double beta,
    double **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dtrmm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        double alpha,
        double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        double **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dtrmm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        double alpha,
        double **dA_array, magma_int_t* ldda,
        double **dB_array, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dtrmm_vbatched_max(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        double alpha,
        double **dA_array, magma_int_t* ldda,
        double **dB_array, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dtrmm_vbatched_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t* m, magma_int_t* n,
        double alpha,
        double **dA_array, magma_int_t* ldda,
        double **dB_array, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dtrmm_vbatched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t* m, magma_int_t* n,
        double alpha,
        double **dA_array, magma_int_t* ldda,
        double **dB_array, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dtrsm_small_vbatched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        double alpha,
        double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        double **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dtrsm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        double alpha,
        double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        double **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dtrsm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        double alpha,
        double **dA_array, magma_int_t* ldda,
        double **dB_array, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dtrsm_vbatched_max(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_dtrsm_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_dtrsm_inv_outofplace_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag,
    magma_int_t *m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    double** dX_array,    magma_int_t* lddx,
    double** dinvA_array, magma_int_t* dinvA_length,
    double** dA_displ, double** dB_displ,
    double** dX_displ, double** dinvA_displ,
    magma_int_t resetozero,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n,
    magma_queue_t queue);

void magmablas_dtrsm_inv_work_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag,
    magma_int_t* m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    double** dX_array,    magma_int_t* lddx,
    double** dinvA_array, magma_int_t* dinvA_length,
    double** dA_displ, double** dB_displ,
    double** dX_displ, double** dinvA_displ,
    magma_int_t resetozero,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n,
    magma_queue_t queue);

void magmablas_dtrsm_inv_vbatched_max_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n,
    magma_queue_t queue);

void
magmablas_dtrsm_inv_vbatched_max(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n,
    magma_queue_t queue);

void
magmablas_dtrsm_inv_vbatched_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount,
    magma_queue_t queue);

void
magmablas_dtrsm_inv_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    double alpha,
    double** dA_array,    magma_int_t* ldda,
    double** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount,
    magma_queue_t queue);

void
magmablas_dtrtri_diag_vbatched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t nmax, magma_int_t *n,
    double const * const *dA_array, magma_int_t *ldda,
    double **dinvA_array,
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);

void
magmablas_dsymm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t *m, magma_int_t *n,
        double alpha,
        double **dA_array, magma_int_t *ldda,
        double **dB_array, magma_int_t *lddb,
        double beta,
        double **dC_array, magma_int_t *lddc,
        magma_int_t max_m, magma_int_t max_n,
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC,
        magma_int_t specM, magma_int_t specN,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsymm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t *m, magma_int_t *n,
        double alpha,
        double **dA_array, magma_int_t *ldda,
        double **dB_array, magma_int_t *lddb,
        double beta,
        double **dC_array, magma_int_t *lddc,
        magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
        magma_queue_t queue );

void
magmablas_dsymm_vbatched_max(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t *m, magma_int_t *n,
        double alpha,
        double **dA_array, magma_int_t *ldda,
        double **dB_array, magma_int_t *lddb,
        double beta,
        double **dC_array, magma_int_t *lddc,
        magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
        magma_queue_t queue );

void
magmablas_dsymm_vbatched_nocheck(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t *m, magma_int_t *n,
        double alpha,
        double **dA_array, magma_int_t *ldda,
        double **dB_array, magma_int_t *lddb,
        double beta,
        double **dC_array, magma_int_t *lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsymm_vbatched(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t *m, magma_int_t *n,
        double alpha,
        double **dA_array, magma_int_t *ldda,
        double **dB_array, magma_int_t *lddb,
        double beta,
        double **dC_array, magma_int_t *lddc,
        magma_int_t batchCount, magma_queue_t queue );

/* Level 2 */
void
magmablas_dgemv_vbatched_max_nocheck(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n,
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda,
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);

void
magmablas_dgemv_vbatched_max(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n,
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda,
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);

void
magmablas_dgemv_vbatched_nocheck(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n,
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda,
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_dgemv_vbatched(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n,
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda,
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_dsymv_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_int_t* n, double alpha,
    double **dA_array, magma_int_t* ldda,
    double **dX_array, magma_int_t* incx,
    double beta,
    double **dY_array, magma_int_t* incy,
    magma_int_t max_n, magma_int_t batchCount, magma_queue_t queue );

void
magmablas_dsymv_vbatched_max(
    magma_uplo_t uplo, magma_int_t* n,
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda,
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount,
    magma_int_t max_n, magma_queue_t queue);

void
magmablas_dsymv_vbatched_nocheck(
    magma_uplo_t uplo, magma_int_t* n,
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda,
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_dsymv_vbatched(
    magma_uplo_t uplo, magma_int_t* n,
    double alpha,
    magmaDouble_ptr dA_array[], magma_int_t* ldda,
    magmaDouble_ptr dx_array[], magma_int_t* incx,
    double beta,
    magmaDouble_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount, magma_queue_t queue);
/* Level 1 */
/* Auxiliary routines */
void magma_dset_pointer_var_cc(
    double **output_array,
    double *input,
    magma_int_t *lda,
    magma_int_t row, magma_int_t column,
    magma_int_t *batch_offset,
    magma_int_t batchCount,
    magma_queue_t queue);

void
magma_ddisplace_pointers_var_cc(double **output_array,
    double **input_array, magma_int_t* lda,
    magma_int_t row, magma_int_t column,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_ddisplace_pointers_var_cv(double **output_array,
    double **input_array, magma_int_t* lda,
    magma_int_t row, magma_int_t* column,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_ddisplace_pointers_var_vc(double **output_array,
    double **input_array, magma_int_t* lda,
    magma_int_t *row, magma_int_t column,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_ddisplace_pointers_var_vv(double **output_array,
    double **input_array, magma_int_t* lda,
    magma_int_t* row, magma_int_t* column,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_dlaset_vbatched(
    magma_uplo_t uplo, magma_int_t max_m, magma_int_t max_n,
    magma_int_t* m, magma_int_t* n,
    double offdiag, double diag,
    magmaDouble_ptr dAarray[], magma_int_t* ldda,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_dlacpy_vbatched(
    magma_uplo_t uplo,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t* m, magma_int_t* n,
    double const * const * dAarray, magma_int_t* ldda,
    double**               dBarray, magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue );

  /*
   *  Aux. vbatched routines
   */
magma_int_t magma_get_dpotrf_vbatched_crossover();

#ifdef __cplusplus
}
#endif

#undef MAGMA_REAL

#endif  /* MAGMA_DVBATCHED_H */
