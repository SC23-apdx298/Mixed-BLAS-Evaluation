/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @generated from magmablas/zhemm_vbatched_core.cu, normal z -> d, Wed Feb 22 15:21:30 2023

       @author Ahmad Abdelfattah
       
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

#define PRECISION_d
#include "hemm_template_kernel_vbatched.cuh"
/******************************************************************************/
extern "C" void 
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
        magma_int_t batchCount, magma_queue_t queue )
{        
    if(side == MagmaLeft){
        hemm_template_vbatched<double, DSYMM_BATCHED_LEFT>(
            side, uplo, m, n, 
            dA_array, ldda,
            dB_array, lddb, 
            dC_array, lddc, alpha, beta, 
            max_m, max_n, 
            roffA, coffA, roffB, coffB, roffC, coffC, specM, specN, 
            batchCount, queue);
    }else{
        hemm_template_vbatched<double, DSYMM_BATCHED_RIGHT>(
            side, uplo, m, n, 
            dA_array, ldda,
            dB_array, lddb, 
            dC_array, lddc, alpha, beta, 
            max_m, max_n, 
            roffA, coffA, roffB, coffB, roffC, coffC, specM, specN, 
            batchCount, queue);
    }
}

/******************************************************************************/
