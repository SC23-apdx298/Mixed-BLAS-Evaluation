/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @author Hartwig Anzt

       @generated from sparse/blas/zilu.cpp, normal z -> d, Wed Feb 22 15:21:39 2023
*/
#include "magmasparse_internal.h"
#include <cuda.h>  // for CUDA_VERSION

#include "magma_trisolve.h"

#define PRECISION_d

/* For hipSPARSE, they use a separate real type than for hipBLAS */
#if defined(MAGMA_HAVE_HIP)
  #ifdef PRECISION_z
    #define double double
  #elif defined(PRECISION_c)
    #define hipblasComplex hipComplex
  #endif
#endif

#if CUDA_VERSION >= 12000
   #define cusparseCreateCsrsm2Info(info)
   #define cusparseDestroyCsrsm2Info(info)
#endif

// todo: make it spacific
#if CUDA_VERSION >= 11000 || defined(MAGMA_HAVE_HIP)
#define cusparseCreateSolveAnalysisInfo(info) cusparseCreateCsrsm2Info(info) 
#else
#define cusparseCreateSolveAnalysisInfo(info)                                                   \
        CHECK_CUSPARSE( cusparseCreateSolveAnalysisInfo( info ))
#endif

#if CUDA_VERSION >= 11000 || defined(MAGMA_HAVE_HIP)
#define cusparseDestroySolveAnalysisInfo(info) cusparseDestroyCsrsm2Info(info)
#endif

// todo: check the info and linfo if we have to give it back; free memory? 
#if CUDA_VERSION >= 11000
#define cusparseDcsrsm_analysis(handle, op, rows, nnz, descrA, dval, drow, dcol, info )         \
    {                                                                                           \
        double alpha = MAGMA_D_ONE;                                                 \
        double *B;                                                                     \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_GENERAL );                             \
        cusparseDcsrsm2_bufferSizeExt(handle, 0, op, CUSPARSE_OPERATION_NON_TRANSPOSE,          \
                                      rows, 1, nnz, (const double *)&alpha,            \
                                      descrA, dval, drow, dcol,                                 \
                                      B, rows, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, &bufsize); \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        cusparseDcsrsm2_analysis(handle, 0, op, CUSPARSE_OPERATION_NON_TRANSPOSE,               \
                                 rows, 1, nnz, (const double *)&alpha,                 \
                                 descrA, dval, drow, dcol,                                      \
                                 B, rows, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buf);           \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }

#elif defined(MAGMA_HAVE_HIP)
#define cusparseDcsrsm_analysis(handle, op, rows, nnz, descrA, dval, drow, dcol, info )         \
    {                                                                                           \
        double alpha = MAGMA_D_ONE;                                                 \
        double *B;                                                                     \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        hipsparseDcsrsm2_bufferSizeExt(handle, 0, op, HIPSPARSE_OPERATION_NON_TRANSPOSE,         \
                                      rows, 1, nnz, (const double *)&alpha,           \
                                      descrA, (const double *)dval, (const int *)drow, (const int *)dcol,  \
                                      (const double *)B, rows, info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, &bufsize); \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        hipsparseDcsrsm2_analysis(handle, 0, op, HIPSPARSE_OPERATION_NON_TRANSPOSE,               \
                                 rows, 1, nnz, (const double *)&alpha,                 \
                                 descrA, (const double *)dval, drow, dcol,            \
                                 B, rows, info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, buf);           \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }



#endif

#if CUDA_VERSION >= 11000
#define cusparseDcsr2csc(handle, cols, rows, nnz, dval, drow, dcol, prdval, prdcol, prdrow,     \
                         action, base)                                                          \
    {                                                                                           \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        cusparseCsr2cscEx2_bufferSize(handle, cols, rows, nnz, dval, drow, dcol, prdval,        \
                                      prdcol, prdrow, CUDA_R_64F, action, base,                 \
                                      CUSPARSE_CSR2CSC_ALG1, &bufsize);                         \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        cusparseCsr2cscEx2(handle, cols, rows, nnz, dval, drow, dcol, prdval, prdcol, prdrow,   \
                           CUDA_R_64F, action, base, CUSPARSE_CSR2CSC_ALG1, buf);               \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }
#endif

// todo: info is passed from analysis; to change info with this linfo & remove linfo from here
#if CUDA_VERSION >= 11000
#define cusparseDcsric0(handle, op, rows, nnz, descrA, dval, drow, dcol, info )                 \
    {                                                                                           \
        int bufsize;                                                                            \
        void *buf;                                                                              \
        csric02Info_t linfo;                                                                    \
        cusparseCreateCsric02Info(&linfo);                                                      \
        cusparseDcsric02_bufferSize(handle, rows, nnz, descrA, dval, drow, dcol,linfo,&bufsize);\
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        cusparseDcsric02_analysis(handle, rows, nnz, descrA, dval, drow, dcol, linfo,           \
                                  CUSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                         \
        int numerical_zero;                                                                     \
        if (CUSPARSE_STATUS_ZERO_PIVOT ==                                                       \
            cusparseXcsric02_zeroPivot( handle, linfo, &numerical_zero ))                       \
            printf("A(%d,%d) is missing\n", numerical_zero, numerical_zero);                    \
        cusparseDcsric02(handle, rows, nnz, descrA, dval, drow, dcol, linfo,                    \
                         CUSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                                  \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }
#elif defined(MAGMA_HAVE_HIP)
#define cusparseDcsric0(handle, op, rows, nnz, descrA, dval, drow, dcol, info )                 \
    {                                                                                           \
        int bufsize;                                                                            \
        void *buf;                                                                              \
        csric02Info_t linfo;                                                                    \
        hipsparseCreateCsric02Info(&linfo);                                                      \
        hipsparseDcsric02_bufferSize(handle, rows, nnz, descrA, (double*)dval, drow, dcol,linfo,&bufsize);\
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        hipsparseDcsric02_analysis(handle, rows, nnz, descrA, (double*)dval, drow, dcol, linfo,           \
                                  HIPSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                         \
        int numerical_zero;                                                                     \
        if (HIPSPARSE_STATUS_ZERO_PIVOT ==                                                       \
            hipsparseXcsric02_zeroPivot( handle, linfo, &numerical_zero ))                       \
            printf("A(%d,%d) is missing\n", numerical_zero, numerical_zero);                    \
        hipsparseDcsric02(handle, rows, nnz, descrA, (double*)dval, drow, dcol, linfo,                    \
                         HIPSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                                  \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    } 
#else
#define cusparseDcsric0(handle, op, rows, nnz, descrA, dval, drow, dcol, info )                 \
    CHECK_CUSPARSE( cusparseDcsric0(handle, op, rows, descrA, dval, drow, dcol, info ))
#endif

/**
    Purpose
    -------

    Prepares the ILU preconditioner via the cuSPARSE.

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input matrix A

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/

extern "C" magma_int_t
magma_dcumilusetup(
    magma_d_matrix A,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrA=NULL;
#if CUDA_VERSION >= 7000 || defined(MAGMA_HAVE_HIP)
    csrilu02Info_t info_M=NULL;
    void *pBuffer = NULL;
#endif
    
    // magma_dprint_matrix(A, queue );
    // copy matrix into preconditioner parameter
    magma_d_matrix hA={Magma_CSR}, hACSR={Magma_CSR};
    magma_d_matrix hL={Magma_CSR}, hU={Magma_CSR};
    CHECK( magma_dmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
    CHECK( magma_dmconvert( hA, &hACSR, hA.storage_type, Magma_CSR, queue ));

    // in case using fill-in
    if( precond->levels > 0 ){
        magma_d_matrix hAL={Magma_CSR}, hAUt={Magma_CSR};
        CHECK( magma_dsymbilu( &hACSR, precond->levels, &hAL, &hAUt,  queue ));
        magma_dmfree(&hAL, queue);
        magma_dmfree(&hAUt, queue);
    }

    CHECK( magma_dmtransfer(hACSR, &(precond->M), Magma_CPU, Magma_DEV, queue ));

    magma_dmfree( &hA, queue );
    magma_dmfree( &hACSR, queue );

    // CUSPARSE context //
    CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
    CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrA ));
    CHECK_CUSPARSE( cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_GENERAL ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrA, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrA, CUSPARSE_INDEX_BASE_ZERO ));
    cusparseCreateSolveAnalysisInfo( &(precond->cuinfoILU) );

    // use kernel to manually check for zeros n the diagonal
    CHECK( magma_ddiagcheck( precond->M, queue ) );
    
#if CUDA_VERSION >= 7000 
    // this version has the bug fixed where a zero on the diagonal causes a crash
    CHECK_CUSPARSE( cusparseCreateCsrilu02Info(&info_M) );
    int buffersize;
    int structural_zero;
    int numerical_zero;
    
    CHECK_CUSPARSE(
    cusparseDcsrilu02_bufferSize( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M,
                         &buffersize ) );
    
    CHECK( magma_malloc((void**)&pBuffer, buffersize) );

    CHECK_CUSPARSE( cusparseDcsrilu02_analysis( cusparseHandle,
            precond->M.num_rows, precond->M.nnz, descrA,
            precond->M.dval, precond->M.drow, precond->M.dcol,
            info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer ));
    
    CHECK_CUSPARSE( cusparseXcsrilu02_zeroPivot( cusparseHandle, info_M, &numerical_zero ) );
    CHECK_CUSPARSE( cusparseXcsrilu02_zeroPivot( cusparseHandle, info_M, &structural_zero ) );
    
    CHECK_CUSPARSE(
    cusparseDcsrilu02( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer) );

#elif defined(MAGMA_HAVE_HIP)

    // this version has the bug fixed where a zero on the diagonal causes a crash
    CHECK_CUSPARSE( hipsparseCreateCsrilu02Info(&info_M) );
    int buffersize;
    int structural_zero;
    int numerical_zero;
    
    CHECK_CUSPARSE(
    hipsparseDcsrilu02_bufferSize( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         (double*)precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M,
                         &buffersize ) );
    
    CHECK( magma_malloc((void**)&pBuffer, buffersize) );

    CHECK_CUSPARSE( hipsparseDcsrilu02_analysis( cusparseHandle,
            precond->M.num_rows, precond->M.nnz, descrA,
            (double*)precond->M.dval, precond->M.drow, precond->M.dcol,
            info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer ));
    
    CHECK_CUSPARSE( hipsparseXcsrilu02_zeroPivot( cusparseHandle, info_M, &numerical_zero ) );
    CHECK_CUSPARSE( hipsparseXcsrilu02_zeroPivot( cusparseHandle, info_M, &structural_zero ) );
    
    CHECK_CUSPARSE(
    hipsparseDcsrilu02( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         (double*)precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer) );

#else
    // this version contains the bug but is needed for backward compability
    cusparseDcsrsm_analysis( cusparseHandle,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             precond->M.num_rows, precond->M.nnz, descrA,
                             precond->M.dval, precond->M.drow, precond->M.dcol,
                             precond->cuinfoILU );
    CHECK_CUSPARSE( cusparseDcsrilu0( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      precond->M.num_rows, descrA,
                      precond->M.dval,
                      precond->M.drow,
                      precond->M.dcol,
                      precond->cuinfoILU ));
#endif

    CHECK( magma_dmtransfer( precond->M, &hA, Magma_DEV, Magma_CPU, queue ));

    hL.diagorder_type = Magma_UNITY;
    CHECK( magma_dmconvert( hA, &hL , Magma_CSR, Magma_CSRL, queue ));
    hU.diagorder_type = Magma_VALUE;
    CHECK( magma_dmconvert( hA, &hU , Magma_CSR, Magma_CSRU, queue ));
    CHECK( magma_dmtransfer( hL, &(precond->L), Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_dmtransfer( hU, &(precond->U), Magma_CPU, Magma_DEV, queue ));
    
    // malloc aux space for sync-free sptrsv 
    CHECK( magma_index_malloc( &(precond->L_dgraphindegree), precond->M.num_rows ));
    CHECK( magma_index_malloc( &(precond->L_dgraphindegree_bak), precond->M.num_rows ));
    CHECK( magma_index_malloc( &(precond->U_dgraphindegree), precond->M.num_rows ));
    CHECK( magma_index_malloc( &(precond->U_dgraphindegree_bak), precond->M.num_rows ));

    if( precond->trisolver == Magma_CUSOLVE || precond->trisolver == 0 ){
        CHECK(magma_dtrisolve_analysis(precond->L, &precond->cuinfoL, false, false, false, queue));
        CHECK(magma_dtrisolve_analysis(precond->U, &precond->cuinfoU, true, false, false, queue));
    } else if( precond->trisolver == Magma_SYNCFREESOLVE ){
            magma_dmfree(&hL, queue );
            magma_dmfree(&hU, queue );
            magma_dmtransfer( precond->L, &hL, Magma_DEV, Magma_DEV, queue );
            // conversion using CUSPARSE
            #ifdef MAGMA_HAVE_HIP
            hipsparseDcsr2csc(cusparseHandle, hL.num_cols, 
                             hL.num_rows, hL.nnz,
                             (double*)hL.dval, hL.drow, hL.dcol, 
                             (double*)precond->L.dval, precond->L.dcol, precond->L.drow,
                             CUSPARSE_ACTION_NUMERIC,
                             CUSPARSE_INDEX_BASE_ZERO);
            #else
            cusparseDcsr2csc(cusparseHandle, hL.num_cols, 
                             hL.num_rows, hL.nnz,
                             hL.dval, hL.drow, hL.dcol, 
                             precond->L.dval, precond->L.dcol, precond->L.drow,
                             CUSPARSE_ACTION_NUMERIC,
                             CUSPARSE_INDEX_BASE_ZERO);

            #endif

            magma_dmtransfer( precond->U, &hU, Magma_DEV, Magma_DEV, queue );
            // conversion using CUSPARSE

            #ifdef MAGMA_HAVE_HIP
            hipsparseDcsr2csc(cusparseHandle, hU.num_cols, 
                             hU.num_rows, hU.nnz,
                             (double*)hU.dval, hU.drow, hU.dcol, 
                             (double*)precond->U.dval, precond->U.dcol, precond->U.drow,
                             CUSPARSE_ACTION_NUMERIC,
                             CUSPARSE_INDEX_BASE_ZERO);
            #else
            cusparseDcsr2csc(cusparseHandle, hU.num_cols, 
                             hU.num_rows, hU.nnz,
                             hU.dval, hU.drow, hU.dcol, 
                             precond->U.dval, precond->U.dcol, precond->U.drow,
                             CUSPARSE_ACTION_NUMERIC,
                             CUSPARSE_INDEX_BASE_ZERO);
            #endif

            // set this to be CSC
            precond->U.storage_type = Magma_CSC;
            precond->L.storage_type = Magma_CSC;
            
            // analysis sparsity structures of L and U
            magma_dgecscsyncfreetrsm_analysis(precond->L.num_rows, 
                precond->L.nnz, precond->L.dval, 
                precond->L.drow, precond->L.dcol, 
                precond->L_dgraphindegree, precond->L_dgraphindegree_bak, 
                queue);
            magma_dgecscsyncfreetrsm_analysis(precond->U.num_rows, 
                precond->U.nnz, precond->U.dval, 
                precond->U.drow, precond->U.dcol, 
                precond->U_dgraphindegree, precond->U_dgraphindegree_bak, 
                queue);

            magma_dmfree(&hL, queue );
            magma_dmfree(&hU, queue );
    } else {
        //prepare for iterative solves
        
        // extract the diagonal of L into precond->d
        CHECK( magma_djacobisetup_diagscal( precond->L, &precond->d, queue ));
        // precond->d.memory_location = Magma_DEV;
        CHECK( magma_dvinit( &precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_D_ZERO, queue ));
        
        // extract the diagonal of U into precond->d2
        CHECK( magma_djacobisetup_diagscal( precond->U, &precond->d2, queue ));
        // precond->d2.memory_location = Magma_DEV;
        CHECK( magma_dvinit( &precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_D_ZERO, queue ));
    }

    
cleanup:
#if CUDA_VERSION >= 7000 || defined(MAGMA_HAVE_HIP)
    magma_free( pBuffer );
    cusparseDestroyCsrilu02Info( info_M );
#endif
    cusparseDestroySolveAnalysisInfo( precond->cuinfoILU );
    cusparseDestroyMatDescr( descrA );
    cusparseDestroy( cusparseHandle );
    magma_dmfree( &hA, queue );
    magma_dmfree( &hACSR, queue );
    magma_dmfree(&hA, queue );
    magma_dmfree(&hL, queue );
    magma_dmfree(&hU, queue );

    return info;
}



/**
    Purpose
    -------

    Prepares the ILU transpose preconditioner via the cuSPARSE.

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input matrix A

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/

extern "C" magma_int_t
magma_dcumilusetup_transpose(
    magma_d_matrix A,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_d_matrix Ah1={Magma_CSR}, Ah2={Magma_CSR};

    // transpose the matrix
    magma_dmtransfer( precond->L, &Ah1, Magma_DEV, Magma_CPU, queue );
    magma_dmconvert( Ah1, &Ah2, A.storage_type, Magma_CSR, queue );
    magma_dmfree(&Ah1, queue );
    magma_dmtransposeconjugate( Ah2, &Ah1, queue );
    magma_dmfree(&Ah2, queue );
    Ah2.blocksize = A.blocksize;
    Ah2.alignment = A.alignment;
    magma_dmconvert( Ah1, &Ah2, Magma_CSR, A.storage_type, queue );
    magma_dmfree(&Ah1, queue );
    magma_dmtransfer( Ah2, &(precond->LT), Magma_CPU, Magma_DEV, queue );
    magma_dmfree(&Ah2, queue );
    
    magma_dmtransfer( precond->U, &Ah1, Magma_DEV, Magma_CPU, queue );
    magma_dmconvert( Ah1, &Ah2, A.storage_type, Magma_CSR, queue );
    magma_dmfree(&Ah1, queue );
    magma_dmtransposeconjugate( Ah2, &Ah1, queue );
    magma_dmfree(&Ah2, queue );
    Ah2.blocksize = A.blocksize;
    Ah2.alignment = A.alignment;
    magma_dmconvert( Ah1, &Ah2, Magma_CSR, A.storage_type, queue );
    magma_dmfree(&Ah1, queue );
    magma_dmtransfer( Ah2, &(precond->UT), Magma_CPU, Magma_DEV, queue );
    magma_dmfree(&Ah2, queue );
   
    CHECK(magma_dtrisolve_analysis(precond->LT, &precond->cuinfoLT, true, false, false, queue));
    CHECK(magma_dtrisolve_analysis(precond->UT, &precond->cuinfoUT, false, false, false, queue));

cleanup:
    magma_dmfree(&Ah1, queue );
    magma_dmfree(&Ah2, queue );

    return info;
}



/**
    Purpose
    -------

    Prepares the ILU triangular solves via cuSPARSE using an ILU factorization
    matrix stored either in precond->M or on the device as
    precond->L and precond->U.

    Arguments
    ---------

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/

extern "C" magma_int_t
magma_dcumilugeneratesolverinfo(
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_d_matrix hA={Magma_CSR}, hL={Magma_CSR}, hU={Magma_CSR};
    
    if (precond->L.memory_location != Magma_DEV ){
        CHECK( magma_dmtransfer( precond->M, &hA,
        precond->M.memory_location, Magma_CPU, queue ));

        hL.diagorder_type = Magma_UNITY;
        CHECK( magma_dmconvert( hA, &hL , Magma_CSR, Magma_CSRL, queue ));
        hU.diagorder_type = Magma_VALUE;
        CHECK( magma_dmconvert( hA, &hU , Magma_CSR, Magma_CSRU, queue ));
        CHECK( magma_dmtransfer( hL, &(precond->L), Magma_CPU, Magma_DEV, queue ));
        CHECK( magma_dmtransfer( hU, &(precond->U), Magma_CPU, Magma_DEV, queue ));
        
        magma_dmfree(&hA, queue );
        magma_dmfree(&hL, queue );
        magma_dmfree(&hU, queue );
    }
    
    CHECK(magma_dtrisolve_analysis(precond->L, &precond->cuinfoL, false, false, false, queue));
    CHECK(magma_dtrisolve_analysis(precond->U, &precond->cuinfoU, true, false, false, queue));
    
    if( precond->trisolver != 0 && precond->trisolver != Magma_CUSOLVE ){
        //prepare for iterative solves

        // extract the diagonal of L into precond->d
        CHECK( magma_djacobisetup_diagscal( precond->L, &precond->d, queue ));
        CHECK( magma_dvinit( &precond->work1, Magma_DEV, precond->U.num_rows, 1, MAGMA_D_ZERO, queue ));
        
        // extract the diagonal of U into precond->d2
        CHECK( magma_djacobisetup_diagscal( precond->U, &precond->d2, queue ));
        CHECK( magma_dvinit( &precond->work2, Magma_DEV, precond->U.num_rows, 1, MAGMA_D_ZERO, queue ));
    }
    
cleanup:     
    return info;
}


/**
    Purpose
    -------

    Performs the left triangular solves using the ILU preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_d_matrix
                RHS

    @param[in,out]
    x           magma_d_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/

extern "C" magma_int_t
magma_dapplycumilu_l(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
        
    double one = MAGMA_D_MAKE( 1.0, 0.0);

    // CUSPARSE context //
    if( precond->trisolver == Magma_CUSOLVE || precond->trisolver == 0 ){
        CHECK(magma_dtrisolve(precond->L, precond->cuinfoL, false, false, false, b, *x, queue));
    } else if( precond->trisolver == Magma_SYNCFREESOLVE ){
        magma_dgecscsyncfreetrsm_solve( precond->L.num_rows,
            precond->L.nnz, 
            precond->L.dval, precond->L.drow, precond->L.dcol, 
            precond->L_dgraphindegree, precond->L_dgraphindegree_bak, 
            x->dval, b.dval, 0, //MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD
            1, // rhs
            queue );
    }
       

cleanup:
    return info;
}



/**
    Purpose
    -------

    Performs the left triangular solves using the transpose ILU preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_d_matrix
                RHS

    @param[in,out]
    x           magma_d_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/
   
extern "C" magma_int_t
magma_dapplycumilu_l_transpose(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    double one = MAGMA_D_MAKE( 1.0, 0.0);

    // CUSPARSE context //
    CHECK(magma_dtrisolve(precond->LT, precond->cuinfoLT, true, false, false, b, *x, queue));
    
    

cleanup:
    return info;
}


/**
    Purpose
    -------

    Performs the right triangular solves using the ILU preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_d_matrix
                RHS

    @param[in,out]
    x           magma_d_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/

extern "C" magma_int_t
magma_dapplycumilu_r(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    double one = MAGMA_D_MAKE( 1.0, 0.0);

    // CUSPARSE context //
    if( precond->trisolver == Magma_CUSOLVE || precond->trisolver == 0 ){
        CHECK(magma_dtrisolve(precond->U, precond->cuinfoU, true, false, false, b, *x, queue));
    } else if( precond->trisolver == Magma_SYNCFREESOLVE ){
        magma_dgecscsyncfreetrsm_solve( precond->U.num_rows,
            precond->U.nnz,
            precond->U.dval, precond->U.drow, precond->U.dcol, 
            precond->U_dgraphindegree, precond->U_dgraphindegree_bak, 
            x->dval, b.dval, 1, //MAGMA_CSC_SYNCFREE_SUBSTITUTION_BACKWARD
            1, // rhs
            queue );
    }
    
    

cleanup:
    return info; 
}


/**
    Purpose
    -------

    Performs the right triangular solves using the transpose ILU preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_d_matrix
                RHS

    @param[in,out]
    x           magma_d_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/

extern "C" magma_int_t
magma_dapplycumilu_r_transpose(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    double one = MAGMA_D_MAKE( 1.0, 0.0);

    CHECK(magma_dtrisolve(precond->UT, precond->cuinfoUT, false, false, false, b, *x, queue));
    
cleanup:
    return info; 
}


/**
    Purpose
    -------

    Prepares the IC preconditioner via cuSPARSE.

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input matrix A

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dhepr
*******************************************************************************/

extern "C" magma_int_t
magma_dcumiccsetup(
    magma_d_matrix A,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrA=NULL;
    cusparseMatDescr_t descrL=NULL;
    cusparseMatDescr_t descrU=NULL;
#if CUDA_VERSION >= 7000
    csric02Info_t info_M=NULL;
    void *pBuffer = NULL;
#endif
    
    magma_d_matrix hA={Magma_CSR}, hACSR={Magma_CSR}, U={Magma_CSR};
    CHECK( magma_dmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
    U.diagorder_type = Magma_VALUE;
    CHECK( magma_dmconvert( hA, &hACSR, hA.storage_type, Magma_CSR, queue ));

    // in case using fill-in
    if( precond->levels > 0 ){
            magma_d_matrix hAL={Magma_CSR}, hAUt={Magma_CSR};
            CHECK( magma_dsymbilu( &hACSR, precond->levels, &hAL, &hAUt,  queue ));
            magma_dmfree(&hAL, queue);
            magma_dmfree(&hAUt, queue);
    }

    CHECK( magma_dmconvert( hACSR, &U, Magma_CSR, Magma_CSRL, queue ));
    magma_dmfree( &hACSR, queue );
    CHECK( magma_dmtransfer(U, &(precond->M), Magma_CPU, Magma_DEV, queue ));

    // CUSPARSE context //
    CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
    CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrA ));
    cusparseCreateSolveAnalysisInfo( &(precond->cuinfoILU) );
    // use kernel to manually check for zeros n the diagonal
    CHECK( magma_ddiagcheck( precond->M, queue ) );
    
#if CUDA_VERSION >= 12000
    // this version has the bug fixed where a zero on the diagonal causes a crash
    CHECK_CUSPARSE( cusparseCreateCsric02Info(&info_M) );
    CHECK_CUSPARSE( cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_GENERAL ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrA, CUSPARSE_INDEX_BASE_ZERO ));
    int buffersize;
    int structural_zero;
    int numerical_zero;
    
    CHECK_CUSPARSE(
    cusparseDcsric02_bufferSize( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M,
                         &buffersize ) );
    
    CHECK( magma_malloc((void**)&pBuffer, buffersize) );

    CHECK_CUSPARSE( cusparseDcsric02_analysis( cusparseHandle,
            precond->M.num_rows, precond->M.nnz, descrA,
            precond->M.dval, precond->M.drow, precond->M.dcol,
            info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer ));
    CHECK_CUSPARSE( cusparseXcsric02_zeroPivot( cusparseHandle, info_M, &numerical_zero ) );
    CHECK_CUSPARSE( cusparseXcsric02_zeroPivot( cusparseHandle, info_M, &structural_zero ) );

    CHECK_CUSPARSE(
    cusparseDcsric02( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer) );    

#else
    // this version contains the bug but is needed for backward compability
    CHECK_CUSPARSE( cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrA, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrA, CUSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE( cusparseSetMatFillMode( descrA, CUSPARSE_FILL_MODE_LOWER ));
    
    // todo: Zcsric0 needs different analysis (cusparseDcsric02_analysis)
    cusparseDcsrsm_analysis( cusparseHandle,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             precond->M.num_rows, precond->M.nnz, descrA,
                             precond->M.dval, precond->M.drow, precond->M.dcol,
                             precond->cuinfoILU );
    cusparseDcsric0( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     precond->M.num_rows, precond->M.nnz, descrA,
                     precond->M.dval,
                     precond->M.drow,
                     precond->M.dcol,
                     precond->cuinfoILU );
#endif

    CHECK( magma_dmtransfer( precond->M, &precond->L, 
        Magma_DEV, Magma_DEV, queue ));
    CHECK( magma_dmtranspose(precond->M, &precond->U, queue ));

    if (precond->trisolver == 0 || precond->trisolver == Magma_CUSOLVE) {
        CHECK(magma_dcumicgeneratesolverinfo(precond, queue));
    } else {
        //prepare for iterative solves

        // extract the diagonal of L into precond->d
        CHECK(magma_djacobisetup_diagscal(precond->L, &precond->d, queue));
        CHECK(magma_dvinit(&precond->work1, Magma_DEV, hA.num_rows, 1, 
            MAGMA_D_ZERO, queue));

        // extract the diagonal of U into precond->d2
        CHECK(magma_djacobisetup_diagscal(precond->U, &precond->d2, queue));
        CHECK(magma_dvinit(&precond->work2, Magma_DEV, hA.num_rows, 1, 
            MAGMA_D_ZERO, queue));
    }



/*
    // to enable also the block-asynchronous iteration for the triangular solves
    CHECK( magma_dmtransfer( precond->M, &hA, Magma_DEV, Magma_CPU, queue ));
    hA.storage_type = Magma_CSR;

    magma_d_matrix hD, hR, hAt

    CHECK( magma_dcsrsplit( 256, hA, &hD, &hR, queue ));

    CHECK( magma_dmtransfer( hD, &precond->LD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_dmtransfer( hR, &precond->L, Magma_CPU, Magma_DEV, queue ));

    magma_dmfree(&hD, queue );
    magma_dmfree(&hR, queue );

    CHECK( magma_d_cucsrtranspose(   hA, &hAt, queue ));

    CHECK( magma_dcsrsplit( 256, hAt, &hD, &hR, queue ));

    CHECK( magma_dmtransfer( hD, &precond->UD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_dmtransfer( hR, &precond->U, Magma_CPU, Magma_DEV, queue ));
    
    magma_dmfree(&hD, queue );
    magma_dmfree(&hR, queue );
    magma_dmfree(&hA, queue );
    magma_dmfree(&hAt, queue );
*/

cleanup:
#if CUDA_VERSION >= 7000
    magma_free( pBuffer );
    cusparseDestroyCsric02Info( info_M );
#endif
    cusparseDestroySolveAnalysisInfo( precond->cuinfoILU );
    cusparseDestroyMatDescr( descrL );
    cusparseDestroyMatDescr( descrU );
    cusparseDestroyMatDescr( descrA );
    cusparseDestroy( cusparseHandle );
    magma_dmfree(&U, queue );
    magma_dmfree(&hA, queue );

    return info;
}


/**
    Purpose
    -------

    Prepares the IC preconditioner solverinfo via cuSPARSE for a triangular
    matrix present on the device in precond->M.

    Arguments
    ---------
    
    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dhepr
    ********************************************************************/

extern "C" magma_int_t
magma_dcumicgeneratesolverinfo(
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    CHECK(magma_dtrisolve_analysis(precond->M, &precond->cuinfoL, false, false, false, queue));
    CHECK(magma_dtrisolve_analysis(precond->M, &precond->cuinfoU, false, false, true, queue));
    

/*
    // to enable also the block-asynchronous iteration for the triangular solves
    CHECK( magma_dmtransfer( precond->M, &hA, Magma_DEV, Magma_CPU, queue ));
    hA.storage_type = Magma_CSR;

    CHECK( magma_dcsrsplit( 256, hA, &hD, &hR, queue ));

    CHECK( magma_dmtransfer( hD, &precond->LD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_dmtransfer( hR, &precond->L, Magma_CPU, Magma_DEV, queue ));

    magma_dmfree(&hD, queue );
    magma_dmfree(&hR, queue );

    CHECK( magma_d_cucsrtranspose(   hA, &hAt, queue ));

    CHECK( magma_dcsrsplit( 256, hAt, &hD, &hR, queue ));

    CHECK( magma_dmtransfer( hD, &precond->UD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_dmtransfer( hR, &precond->U, Magma_CPU, Magma_DEV, queue ));
    
    magma_dmfree(&hD, queue );
    magma_dmfree(&hR, queue );
    magma_dmfree(&hA, queue );
    magma_dmfree(&hAt, queue );
*/

cleanup:
    return info;
}



/**
    Purpose
    -------

    Performs the left triangular solves using the ICC preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_d_matrix
                RHS

    @param[in,out]
    x           magma_d_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dhepr
    ********************************************************************/

extern "C" magma_int_t
magma_dapplycumicc_l(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    double one = MAGMA_D_MAKE( 1.0, 0.0);

    CHECK(magma_dtrisolve(precond->M, precond->cuinfoL, false, false, false, b, *x, queue));

cleanup:
    return info; 
}


/**
    Purpose
    -------

    Performs the right triangular solves using the ICC preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_d_matrix
                RHS

    @param[in,out]
    x           magma_d_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dhepr
    ********************************************************************/

extern "C" magma_int_t
magma_dapplycumicc_r(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    double one = MAGMA_D_MAKE( 1.0, 0.0);

    CHECK(magma_dtrisolve(precond->M, precond->cuinfoU, false, false, true, b, *x, queue));
    
    

cleanup:
    return info; 
}



/**
    Purpose
    -------

    Performs the left triangular solves using the IC preconditioner via Jacobi.

    Arguments
    ---------

    @param[in]
    b           magma_d_matrix
                RHS

    @param[out]
    x           magma_d_matrix*
                vector to precondition

    @param[in]
    precond     magma_d_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_dapplyiteric_l(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t dofs = precond->L.num_rows;
    magma_d_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = precond->maxiter;

    // compute c = D^{-1}b and copy c as initial guess to x
    CHECK( magma_djacobisetup_vector_gpu( dofs, b, precond->d,
                                                precond->work1, x, queue ));
    // Jacobi iterator
    CHECK( magma_djacobiiter_precond( precond->L, x, &jacobiiter_par, precond , queue ));

cleanup:
    return info;
}


/**
    Purpose
    -------

    Performs the right triangular solves using the IC preconditioner via Jacobi.

    Arguments
    ---------

    @param[in]
    b           magma_d_matrix
                RHS

    @param[out]
    x           magma_d_matrix*
                vector to precondition

    @param[in]
    precond     magma_d_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_dapplyiteric_r(
    magma_d_matrix b,
    magma_d_matrix *x,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_int_t dofs = precond->U.num_rows;
    magma_d_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = precond->maxiter;

    // compute c = D^{-1}b and copy c as initial guess to x
    CHECK( magma_djacobisetup_vector_gpu( dofs, b, precond->d,
                                                precond->work1, x, queue ));

    // Jacobi iterator
    CHECK( magma_djacobiiter_precond( precond->U, x, &jacobiiter_par, precond , queue ));
    
cleanup:
    return info;
}

