/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023
       
       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @generated from src/zpotf2_vbatched.cpp, normal z -> c, Wed Feb 22 15:19:47 2023
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

#define PRECISION_c
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_cpotf2_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n, 
    magmaFloatComplex **dA_array, magma_int_t* lda,
    magmaFloatComplex **dA_displ, 
    magmaFloatComplex **dW_displ,
    magmaFloatComplex **dB_displ, 
    magmaFloatComplex **dC_displ, 
    magma_int_t *info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo=0;

    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable \n");
    }
    else{
        arginfo = magma_cpotrf_lpout_vbatched(uplo, n, max_n, dA_array, lda, gbstep, info_array, batchCount, queue);
    }

    return arginfo;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
