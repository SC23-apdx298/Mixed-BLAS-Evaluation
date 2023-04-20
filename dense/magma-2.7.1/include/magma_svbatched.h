/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @generated from include/magma_zvbatched.h, normal z -> s, Wed Feb 22 15:21:05 2023
*/

#ifndef MAGMA_SVBATCHED_H
#define MAGMA_SVBATCHED_H

#include "magma_types.h"

#define MAGMA_REAL

#ifdef __cplusplus
extern "C" {
#endif

  /*
   *  control and tuning
   */
void magma_get_sgetrf_vbatched_nbparam(magma_int_t max_m, magma_int_t max_n, magma_int_t *nb, magma_int_t *recnb);


  /*
   *  LAPACK vbatched routines
   */

magma_int_t
magma_sgetf2_fused_vbatched(
    magma_int_t max_M, magma_int_t max_N,
    magma_int_t max_minMN, magma_int_t max_MxN,
    magma_int_t* M, magma_int_t* N,
    float **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t **dipiv_array, magma_int_t ipiv_i,
    magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue);

magma_int_t
magma_sgetf2_fused_sm_vbatched(
    magma_int_t max_M, magma_int_t max_N, magma_int_t max_minMN, magma_int_t max_MxN,
    magma_int_t* m, magma_int_t* n,
    float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t** dipiv_array, magma_int_t ipiv_i,
    magma_int_t* info_array, magma_int_t gbstep,
    magma_int_t nthreads, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue );

magma_int_t
magma_sgetrf_vbatched(
        magma_int_t* m, magma_int_t* n,
        float **dA_array, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgetrf_vbatched_max_nocheck(
        magma_int_t* m, magma_int_t* n, magma_int_t* minmn,
        magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
        magma_int_t nb, magma_int_t recnb,
        float **dA_array, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t** pivinfo_array,
        magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgetrf_vbatched_max_nocheck_work(
        magma_int_t* m, magma_int_t* n,
        magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
        float **dA_array, magma_int_t *ldda,
        magma_int_t **dipiv_array, magma_int_t *info_array,
        void* work, magma_int_t* lwork,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_isamax_vbatched(
        magma_int_t length, magma_int_t *M, magma_int_t *N,
        float **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t** ipiv_array, magma_int_t ipiv_i,
        magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sswap_vbatched(
        magma_int_t max_n, magma_int_t *M, magma_int_t *N,
        float **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t** ipiv_array, magma_int_t piv_adjustment,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t magma_sscal_sger_vbatched(
    magma_int_t max_M, magma_int_t max_N,
    magma_int_t *M, magma_int_t *N,
    float **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
    magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgetf2_vbatched(
    magma_int_t *m, magma_int_t *n, magma_int_t *minmn,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
    float **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
    magma_int_t **ipiv_array, magma_int_t *info_array,
    magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_sgetrf_recpanel_vbatched(
    magma_int_t* m, magma_int_t* n, magma_int_t* minmn,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn,
    magma_int_t max_mxn, magma_int_t min_recpnb,
    float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t** dipiv_array, magma_int_t dipiv_i, magma_int_t** dpivinfo_array,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount,  magma_queue_t queue);

void
magma_slaswp_left_rowserial_vbatched(
        magma_int_t n,
        magma_int_t *M, magma_int_t *N, float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t ipiv_offset,
        magma_int_t k1, magma_int_t k2,
        magma_int_t batchCount, magma_queue_t queue);

void
magma_slaswp_right_rowserial_vbatched(
        magma_int_t n,
        magma_int_t *M, magma_int_t *N, float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t **ipiv_array, magma_int_t ipiv_offset,
        magma_int_t k1, magma_int_t k2,
        magma_int_t batchCount, magma_queue_t queue);

void
magma_slaswp_left_rowparallel_vbatched(
        magma_int_t n,
        magma_int_t* M, magma_int_t* N,
        float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t k1, magma_int_t k2,
        magma_int_t **pivinfo_array, magma_int_t pivinfo_i,
        magma_int_t batchCount, magma_queue_t queue);

void
magma_slaswp_right_rowparallel_vbatched(
        magma_int_t n,
        magma_int_t* M, magma_int_t* N,
        float** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t k1, magma_int_t k2,
        magma_int_t **pivinfo_array, magma_int_t pivinfo_i,
        magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_spotrf_lpout_vbatched(
    magma_uplo_t uplo, magma_int_t *n, magma_int_t max_n,
    float **dA_array, magma_int_t *lda, magma_int_t gbstep,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_spotf2_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n,
    float **dA_array, magma_int_t* lda,
    float **dA_displ,
    float **dW_displ,
    float **dB_displ,
    float **dC_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_spotrf_panel_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n,
    magma_int_t *ibvec, magma_int_t nb,
    float** dA_array,    magma_int_t* ldda,
    float** dX_array,    magma_int_t* dX_length,
    float** dinvA_array, magma_int_t* dinvA_length,
    float** dW0_displ, float** dW1_displ,
    float** dW2_displ, float** dW3_displ,
    float** dW4_displ,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

magma_int_t
magma_spotrf_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_int_t *n,
    float **dA_array, magma_int_t *ldda,
    magma_int_t *info_array,  magma_int_t batchCount,
    magma_int_t max_n, magma_queue_t queue);

magma_int_t
magma_spotrf_vbatched(
    magma_uplo_t uplo, magma_int_t *n,
    float **dA_array, magma_int_t *ldda,
    magma_int_t *info_array,  magma_int_t batchCount,
    magma_queue_t queue);
  /*
   *  BLAS vbatched routines
   */
/* Level 3 */
void
magmablas_sgemm_vbatched_core(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    float beta,
    float              ** dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sgemm_vbatched_max_nocheck(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sgemm_vbatched_max(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sgemm_vbatched_nocheck(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_sgemm_vbatched(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyrk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc,
    magma_int_t max_n, magma_int_t max_k,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyrk_internal_vbatched(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc,
    magma_int_t max_n, magma_int_t max_k,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyrk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float beta,
    float **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_ssyrk_vbatched_max(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        float alpha,
        float const * const * dA_array, magma_int_t* ldda,
        float beta,
        float **dC_array, magma_int_t* lddc,
        magma_int_t batchCount,
        magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_ssyrk_vbatched_nocheck(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        float alpha,
        float const * const * dA_array, magma_int_t* ldda,
        float beta,
        float **dC_array, magma_int_t* lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyrk_vbatched(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        float alpha,
        float const * const * dA_array, magma_int_t* ldda,
        float beta,
        float **dC_array, magma_int_t* lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyrk_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float beta,
    float **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_ssyrk_vbatched_max(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        float alpha,
        float const * const * dA_array, magma_int_t* ldda,
        float beta,
        float **dC_array, magma_int_t* lddc,
        magma_int_t batchCount,
        magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_ssyrk_vbatched_nocheck(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        float alpha,
        float const * const * dA_array, magma_int_t* ldda,
        float beta,
        float **dC_array, magma_int_t* lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyrk_vbatched(
        magma_uplo_t uplo, magma_trans_t trans,
        magma_int_t* n, magma_int_t* k,
        float alpha,
        float const * const * dA_array, magma_int_t* ldda,
        float beta,
        float **dC_array, magma_int_t* lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyr2k_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta, float **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_ssyr2k_vbatched_max(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta, float **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_ssyr2k_vbatched_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta, float **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyr2k_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta, float **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyr2k_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_ssyr2k_vbatched_max(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc,
    magma_int_t batchCount,
    magma_int_t max_n, magma_int_t max_k, magma_queue_t queue );

void
magmablas_ssyr2k_vbatched_nocheck(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssyr2k_vbatched(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strmm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        float alpha,
        float **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        float **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strmm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        float alpha,
        float **dA_array, magma_int_t* ldda,
        float **dB_array, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strmm_vbatched_max(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        float alpha,
        float **dA_array, magma_int_t* ldda,
        float **dB_array, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strmm_vbatched_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t* m, magma_int_t* n,
        float alpha,
        float **dA_array, magma_int_t* ldda,
        float **dB_array, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strmm_vbatched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t* m, magma_int_t* n,
        float alpha,
        float **dA_array, magma_int_t* ldda,
        float **dB_array, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strsm_small_vbatched(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        float alpha,
        float **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        float **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strsm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        float alpha,
        float **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        float **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strsm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        float alpha,
        float **dA_array, magma_int_t* ldda,
        float **dB_array, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_strsm_vbatched_max(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
    float alpha,
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_strsm_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    float alpha,
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_strsm_inv_outofplace_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag,
    magma_int_t *m, magma_int_t* n,
    float alpha,
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    float** dX_array,    magma_int_t* lddx,
    float** dinvA_array, magma_int_t* dinvA_length,
    float** dA_displ, float** dB_displ,
    float** dX_displ, float** dinvA_displ,
    magma_int_t resetozero,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n,
    magma_queue_t queue);

void magmablas_strsm_inv_work_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t flag,
    magma_int_t* m, magma_int_t* n,
    float alpha,
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    float** dX_array,    magma_int_t* lddx,
    float** dinvA_array, magma_int_t* dinvA_length,
    float** dA_displ, float** dB_displ,
    float** dX_displ, float** dinvA_displ,
    magma_int_t resetozero,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n,
    magma_queue_t queue);

void magmablas_strsm_inv_vbatched_max_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    float alpha,
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n,
    magma_queue_t queue);

void
magmablas_strsm_inv_vbatched_max(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    float alpha,
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n,
    magma_queue_t queue);

void
magmablas_strsm_inv_vbatched_nocheck(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    float alpha,
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount,
    magma_queue_t queue);

void
magmablas_strsm_inv_vbatched(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    float alpha,
    float** dA_array,    magma_int_t* ldda,
    float** dB_array,    magma_int_t* lddb,
    magma_int_t batchCount,
    magma_queue_t queue);

void
magmablas_strtri_diag_vbatched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t nmax, magma_int_t *n,
    float const * const *dA_array, magma_int_t *ldda,
    float **dinvA_array,
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue);

void
magmablas_ssymm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t *m, magma_int_t *n,
        float alpha,
        float **dA_array, magma_int_t *ldda,
        float **dB_array, magma_int_t *lddb,
        float beta,
        float **dC_array, magma_int_t *lddc,
        magma_int_t max_m, magma_int_t max_n,
        magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC,
        magma_int_t specM, magma_int_t specN,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssymm_vbatched_max_nocheck(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t *m, magma_int_t *n,
        float alpha,
        float **dA_array, magma_int_t *ldda,
        float **dB_array, magma_int_t *lddb,
        float beta,
        float **dC_array, magma_int_t *lddc,
        magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
        magma_queue_t queue );

void
magmablas_ssymm_vbatched_max(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t *m, magma_int_t *n,
        float alpha,
        float **dA_array, magma_int_t *ldda,
        float **dB_array, magma_int_t *lddb,
        float beta,
        float **dC_array, magma_int_t *lddc,
        magma_int_t batchCount, magma_int_t max_m, magma_int_t max_n,
        magma_queue_t queue );

void
magmablas_ssymm_vbatched_nocheck(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t *m, magma_int_t *n,
        float alpha,
        float **dA_array, magma_int_t *ldda,
        float **dB_array, magma_int_t *lddb,
        float beta,
        float **dC_array, magma_int_t *lddc,
        magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssymm_vbatched(
        magma_side_t side, magma_uplo_t uplo,
        magma_int_t *m, magma_int_t *n,
        float alpha,
        float **dA_array, magma_int_t *ldda,
        float **dB_array, magma_int_t *lddb,
        float beta,
        float **dC_array, magma_int_t *lddc,
        magma_int_t batchCount, magma_queue_t queue );

/* Level 2 */
void
magmablas_sgemv_vbatched_max_nocheck(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n,
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t* ldda,
    magmaFloat_ptr dx_array[], magma_int_t* incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);

void
magmablas_sgemv_vbatched_max(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n,
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t* ldda,
    magmaFloat_ptr dx_array[], magma_int_t* incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount,
    magma_int_t max_m, magma_int_t max_n, magma_queue_t queue);

void
magmablas_sgemv_vbatched_nocheck(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n,
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t* ldda,
    magmaFloat_ptr dx_array[], magma_int_t* incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_sgemv_vbatched(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n,
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t* ldda,
    magmaFloat_ptr dx_array[], magma_int_t* incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_ssymv_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_int_t* n, float alpha,
    float **dA_array, magma_int_t* ldda,
    float **dX_array, magma_int_t* incx,
    float beta,
    float **dY_array, magma_int_t* incy,
    magma_int_t max_n, magma_int_t batchCount, magma_queue_t queue );

void
magmablas_ssymv_vbatched_max(
    magma_uplo_t uplo, magma_int_t* n,
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t* ldda,
    magmaFloat_ptr dx_array[], magma_int_t* incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount,
    magma_int_t max_n, magma_queue_t queue);

void
magmablas_ssymv_vbatched_nocheck(
    magma_uplo_t uplo, magma_int_t* n,
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t* ldda,
    magmaFloat_ptr dx_array[], magma_int_t* incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_ssymv_vbatched(
    magma_uplo_t uplo, magma_int_t* n,
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t* ldda,
    magmaFloat_ptr dx_array[], magma_int_t* incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t* incy,
    magma_int_t batchCount, magma_queue_t queue);
/* Level 1 */
/* Auxiliary routines */
void magma_sset_pointer_var_cc(
    float **output_array,
    float *input,
    magma_int_t *lda,
    magma_int_t row, magma_int_t column,
    magma_int_t *batch_offset,
    magma_int_t batchCount,
    magma_queue_t queue);

void
magma_sdisplace_pointers_var_cc(float **output_array,
    float **input_array, magma_int_t* lda,
    magma_int_t row, magma_int_t column,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_sdisplace_pointers_var_cv(float **output_array,
    float **input_array, magma_int_t* lda,
    magma_int_t row, magma_int_t* column,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_sdisplace_pointers_var_vc(float **output_array,
    float **input_array, magma_int_t* lda,
    magma_int_t *row, magma_int_t column,
    magma_int_t batchCount, magma_queue_t queue);

void
magma_sdisplace_pointers_var_vv(float **output_array,
    float **input_array, magma_int_t* lda,
    magma_int_t* row, magma_int_t* column,
    magma_int_t batchCount, magma_queue_t queue);

void magmablas_slaset_vbatched(
    magma_uplo_t uplo, magma_int_t max_m, magma_int_t max_n,
    magma_int_t* m, magma_int_t* n,
    float offdiag, float diag,
    magmaFloat_ptr dAarray[], magma_int_t* ldda,
    magma_int_t batchCount, magma_queue_t queue);

void
magmablas_slacpy_vbatched(
    magma_uplo_t uplo,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t* m, magma_int_t* n,
    float const * const * dAarray, magma_int_t* ldda,
    float**               dBarray, magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue );

  /*
   *  Aux. vbatched routines
   */
magma_int_t magma_get_spotrf_vbatched_crossover();

#ifdef __cplusplus
}
#endif

#undef MAGMA_REAL

#endif  /* MAGMA_SVBATCHED_H */
