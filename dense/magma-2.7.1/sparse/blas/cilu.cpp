/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @author Hartwig Anzt

       @generated from sparse/blas/zilu.cpp, normal z -> c, Wed Feb 22 15:21:39 2023
*/
#include "magmasparse_internal.h"
#include <cuda.h>  // for CUDA_VERSION

#include "magma_trisolve.h"

#define PRECISION_c

/* For hipSPARSE, they use a separate complex type than for hipBLAS */
#if defined(MAGMA_HAVE_HIP)
  #ifdef PRECISION_z
    #define hipblasComplex hipFloatComplex
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
#define cusparseCcsrsm_analysis(handle, op, rows, nnz, descrA, dval, drow, dcol, info )         \
    {                                                                                           \
        magmaFloatComplex alpha = MAGMA_C_ONE;                                                 \
        cuFloatComplex *B;                                                                     \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_GENERAL );                             \
        cusparseCcsrsm2_bufferSizeExt(handle, 0, op, CUSPARSE_OPERATION_NON_TRANSPOSE,          \
                                      rows, 1, nnz, (const cuFloatComplex *)&alpha,            \
                                      descrA, dval, drow, dcol,                                 \
                                      B, rows, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, &bufsize); \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        cusparseCcsrsm2_analysis(handle, 0, op, CUSPARSE_OPERATION_NON_TRANSPOSE,               \
                                 rows, 1, nnz, (const cuFloatComplex *)&alpha,                 \
                                 descrA, dval, drow, dcol,                                      \
                                 B, rows, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buf);           \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }

#elif defined(MAGMA_HAVE_HIP)
#define cusparseCcsrsm_analysis(handle, op, rows, nnz, descrA, dval, drow, dcol, info )         \
    {                                                                                           \
        magmaFloatComplex alpha = MAGMA_C_ONE;                                                 \
        hipFloatComplex *B;                                                                     \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        hipsparseCcsrsm2_bufferSizeExt(handle, 0, op, HIPSPARSE_OPERATION_NON_TRANSPOSE,         \
                                      rows, 1, nnz, (const hipFloatComplex *)&alpha,           \
                                      descrA, (const hipFloatComplex *)dval, (const int *)drow, (const int *)dcol,  \
                                      (const hipFloatComplex *)B, rows, info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, &bufsize); \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        hipsparseCcsrsm2_analysis(handle, 0, op, HIPSPARSE_OPERATION_NON_TRANSPOSE,               \
                                 rows, 1, nnz, (const hipFloatComplex *)&alpha,                 \
                                 descrA, (const hipFloatComplex *)dval, drow, dcol,            \
                                 B, rows, info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, buf);           \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }



#endif

#if CUDA_VERSION >= 11000
#define cusparseCcsr2csc(handle, cols, rows, nnz, dval, drow, dcol, prdval, prdcol, prdrow,     \
                         action, base)                                                          \
    {                                                                                           \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        cusparseCsr2cscEx2_bufferSize(handle, cols, rows, nnz, dval, drow, dcol, prdval,        \
                                      prdcol, prdrow, CUDA_C_32F, action, base,                 \
                                      CUSPARSE_CSR2CSC_ALG1, &bufsize);                         \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        cusparseCsr2cscEx2(handle, cols, rows, nnz, dval, drow, dcol, prdval, prdcol, prdrow,   \
                           CUDA_C_32F, action, base, CUSPARSE_CSR2CSC_ALG1, buf);               \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }
#endif

// todo: info is passed from analysis; to change info with this linfo & remove linfo from here
#if CUDA_VERSION >= 11000
#define cusparseCcsric0(handle, op, rows, nnz, descrA, dval, drow, dcol, info )                 \
    {                                                                                           \
        int bufsize;                                                                            \
        void *buf;                                                                              \
        csric02Info_t linfo;                                                                    \
        cusparseCreateCsric02Info(&linfo);                                                      \
        cusparseCcsric02_bufferSize(handle, rows, nnz, descrA, dval, drow, dcol,linfo,&bufsize);\
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        cusparseCcsric02_analysis(handle, rows, nnz, descrA, dval, drow, dcol, linfo,           \
                                  CUSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                         \
        int numerical_zero;                                                                     \
        if (CUSPARSE_STATUS_ZERO_PIVOT ==                                                       \
            cusparseXcsric02_zeroPivot( handle, linfo, &numerical_zero ))                       \
            printf("A(%d,%d) is missing\n", numerical_zero, numerical_zero);                    \
        cusparseCcsric02(handle, rows, nnz, descrA, dval, drow, dcol, linfo,                    \
                         CUSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                                  \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }
#elif defined(MAGMA_HAVE_HIP)
#define cusparseCcsric0(handle, op, rows, nnz, descrA, dval, drow, dcol, info )                 \
    {                                                                                           \
        int bufsize;                                                                            \
        void *buf;                                                                              \
        csric02Info_t linfo;                                                                    \
        hipsparseCreateCsric02Info(&linfo);                                                      \
        hipsparseCcsric02_bufferSize(handle, rows, nnz, descrA, (hipFloatComplex*)dval, drow, dcol,linfo,&bufsize);\
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        hipsparseCcsric02_analysis(handle, rows, nnz, descrA, (hipFloatComplex*)dval, drow, dcol, linfo,           \
                                  HIPSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                         \
        int numerical_zero;                                                                     \
        if (HIPSPARSE_STATUS_ZERO_PIVOT ==                                                       \
            hipsparseXcsric02_zeroPivot( handle, linfo, &numerical_zero ))                       \
            printf("A(%d,%d) is missing\n", numerical_zero, numerical_zero);                    \
        hipsparseCcsric02(handle, rows, nnz, descrA, (hipFloatComplex*)dval, drow, dcol, linfo,                    \
                         HIPSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                                  \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    } 
#else
#define cusparseCcsric0(handle, op, rows, nnz, descrA, dval, drow, dcol, info )                 \
    CHECK_CUSPARSE( cusparseCcsric0(handle, op, rows, descrA, dval, drow, dcol, info ))
#endif

/**
    Purpose
    -------

    Prepares the ILU preconditioner via the cuSPARSE.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                input matrix A

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/

extern "C" magma_int_t
magma_ccumilusetup(
    magma_c_matrix A,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrA=NULL;
#if CUDA_VERSION >= 7000 || defined(MAGMA_HAVE_HIP)
    csrilu02Info_t info_M=NULL;
    void *pBuffer = NULL;
#endif
    
    // magma_cprint_matrix(A, queue );
    // copy matrix into preconditioner parameter
    magma_c_matrix hA={Magma_CSR}, hACSR={Magma_CSR};
    magma_c_matrix hL={Magma_CSR}, hU={Magma_CSR};
    CHECK( magma_cmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
    CHECK( magma_cmconvert( hA, &hACSR, hA.storage_type, Magma_CSR, queue ));

    // in case using fill-in
    if( precond->levels > 0 ){
        magma_c_matrix hAL={Magma_CSR}, hAUt={Magma_CSR};
        CHECK( magma_csymbilu( &hACSR, precond->levels, &hAL, &hAUt,  queue ));
        magma_cmfree(&hAL, queue);
        magma_cmfree(&hAUt, queue);
    }

    CHECK( magma_cmtransfer(hACSR, &(precond->M), Magma_CPU, Magma_DEV, queue ));

    magma_cmfree( &hA, queue );
    magma_cmfree( &hACSR, queue );

    // CUSPARSE context //
    CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
    CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrA ));
    CHECK_CUSPARSE( cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_GENERAL ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrA, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrA, CUSPARSE_INDEX_BASE_ZERO ));
    cusparseCreateSolveAnalysisInfo( &(precond->cuinfoILU) );

    // use kernel to manually check for zeros n the diagonal
    CHECK( magma_cdiagcheck( precond->M, queue ) );
    
#if CUDA_VERSION >= 7000 
    // this version has the bug fixed where a zero on the diagonal causes a crash
    CHECK_CUSPARSE( cusparseCreateCsrilu02Info(&info_M) );
    int buffersize;
    int structural_zero;
    int numerical_zero;
    
    CHECK_CUSPARSE(
    cusparseCcsrilu02_bufferSize( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M,
                         &buffersize ) );
    
    CHECK( magma_malloc((void**)&pBuffer, buffersize) );

    CHECK_CUSPARSE( cusparseCcsrilu02_analysis( cusparseHandle,
            precond->M.num_rows, precond->M.nnz, descrA,
            precond->M.dval, precond->M.drow, precond->M.dcol,
            info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer ));
    
    CHECK_CUSPARSE( cusparseXcsrilu02_zeroPivot( cusparseHandle, info_M, &numerical_zero ) );
    CHECK_CUSPARSE( cusparseXcsrilu02_zeroPivot( cusparseHandle, info_M, &structural_zero ) );
    
    CHECK_CUSPARSE(
    cusparseCcsrilu02( cusparseHandle,
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
    hipsparseCcsrilu02_bufferSize( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         (hipFloatComplex*)precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M,
                         &buffersize ) );
    
    CHECK( magma_malloc((void**)&pBuffer, buffersize) );

    CHECK_CUSPARSE( hipsparseCcsrilu02_analysis( cusparseHandle,
            precond->M.num_rows, precond->M.nnz, descrA,
            (hipFloatComplex*)precond->M.dval, precond->M.drow, precond->M.dcol,
            info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer ));
    
    CHECK_CUSPARSE( hipsparseXcsrilu02_zeroPivot( cusparseHandle, info_M, &numerical_zero ) );
    CHECK_CUSPARSE( hipsparseXcsrilu02_zeroPivot( cusparseHandle, info_M, &structural_zero ) );
    
    CHECK_CUSPARSE(
    hipsparseCcsrilu02( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         (hipFloatComplex*)precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer) );

#else
    // this version contains the bug but is needed for backward compability
    cusparseCcsrsm_analysis( cusparseHandle,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             precond->M.num_rows, precond->M.nnz, descrA,
                             precond->M.dval, precond->M.drow, precond->M.dcol,
                             precond->cuinfoILU );
    CHECK_CUSPARSE( cusparseCcsrilu0( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      precond->M.num_rows, descrA,
                      precond->M.dval,
                      precond->M.drow,
                      precond->M.dcol,
                      precond->cuinfoILU ));
#endif

    CHECK( magma_cmtransfer( precond->M, &hA, Magma_DEV, Magma_CPU, queue ));

    hL.diagorder_type = Magma_UNITY;
    CHECK( magma_cmconvert( hA, &hL , Magma_CSR, Magma_CSRL, queue ));
    hU.diagorder_type = Magma_VALUE;
    CHECK( magma_cmconvert( hA, &hU , Magma_CSR, Magma_CSRU, queue ));
    CHECK( magma_cmtransfer( hL, &(precond->L), Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_cmtransfer( hU, &(precond->U), Magma_CPU, Magma_DEV, queue ));
    
    // malloc aux space for sync-free sptrsv 
    CHECK( magma_index_malloc( &(precond->L_dgraphindegree), precond->M.num_rows ));
    CHECK( magma_index_malloc( &(precond->L_dgraphindegree_bak), precond->M.num_rows ));
    CHECK( magma_index_malloc( &(precond->U_dgraphindegree), precond->M.num_rows ));
    CHECK( magma_index_malloc( &(precond->U_dgraphindegree_bak), precond->M.num_rows ));

    if( precond->trisolver == Magma_CUSOLVE || precond->trisolver == 0 ){
        CHECK(magma_ctrisolve_analysis(precond->L, &precond->cuinfoL, false, false, false, queue));
        CHECK(magma_ctrisolve_analysis(precond->U, &precond->cuinfoU, true, false, false, queue));
    } else if( precond->trisolver == Magma_SYNCFREESOLVE ){
            magma_cmfree(&hL, queue );
            magma_cmfree(&hU, queue );
            magma_cmtransfer( precond->L, &hL, Magma_DEV, Magma_DEV, queue );
            // conversion using CUSPARSE
            #ifdef MAGMA_HAVE_HIP
            hipsparseCcsr2csc(cusparseHandle, hL.num_cols, 
                             hL.num_rows, hL.nnz,
                             (hipFloatComplex*)hL.dval, hL.drow, hL.dcol, 
                             (hipFloatComplex*)precond->L.dval, precond->L.dcol, precond->L.drow,
                             CUSPARSE_ACTION_NUMERIC,
                             CUSPARSE_INDEX_BASE_ZERO);
            #else
            cusparseCcsr2csc(cusparseHandle, hL.num_cols, 
                             hL.num_rows, hL.nnz,
                             hL.dval, hL.drow, hL.dcol, 
                             precond->L.dval, precond->L.dcol, precond->L.drow,
                             CUSPARSE_ACTION_NUMERIC,
                             CUSPARSE_INDEX_BASE_ZERO);

            #endif

            magma_cmtransfer( precond->U, &hU, Magma_DEV, Magma_DEV, queue );
            // conversion using CUSPARSE

            #ifdef MAGMA_HAVE_HIP
            hipsparseCcsr2csc(cusparseHandle, hU.num_cols, 
                             hU.num_rows, hU.nnz,
                             (hipFloatComplex*)hU.dval, hU.drow, hU.dcol, 
                             (hipFloatComplex*)precond->U.dval, precond->U.dcol, precond->U.drow,
                             CUSPARSE_ACTION_NUMERIC,
                             CUSPARSE_INDEX_BASE_ZERO);
            #else
            cusparseCcsr2csc(cusparseHandle, hU.num_cols, 
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
            magma_cgecscsyncfreetrsm_analysis(precond->L.num_rows, 
                precond->L.nnz, precond->L.dval, 
                precond->L.drow, precond->L.dcol, 
                precond->L_dgraphindegree, precond->L_dgraphindegree_bak, 
                queue);
            magma_cgecscsyncfreetrsm_analysis(precond->U.num_rows, 
                precond->U.nnz, precond->U.dval, 
                precond->U.drow, precond->U.dcol, 
                precond->U_dgraphindegree, precond->U_dgraphindegree_bak, 
                queue);

            magma_cmfree(&hL, queue );
            magma_cmfree(&hU, queue );
    } else {
        //prepare for iterative solves
        
        // extract the diagonal of L into precond->d
        CHECK( magma_cjacobisetup_diagscal( precond->L, &precond->d, queue ));
        // precond->d.memory_location = Magma_DEV;
        CHECK( magma_cvinit( &precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_C_ZERO, queue ));
        
        // extract the diagonal of U into precond->d2
        CHECK( magma_cjacobisetup_diagscal( precond->U, &precond->d2, queue ));
        // precond->d2.memory_location = Magma_DEV;
        CHECK( magma_cvinit( &precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_C_ZERO, queue ));
    }

    
cleanup:
#if CUDA_VERSION >= 7000 || defined(MAGMA_HAVE_HIP)
    magma_free( pBuffer );
    cusparseDestroyCsrilu02Info( info_M );
#endif
    cusparseDestroySolveAnalysisInfo( precond->cuinfoILU );
    cusparseDestroyMatDescr( descrA );
    cusparseDestroy( cusparseHandle );
    magma_cmfree( &hA, queue );
    magma_cmfree( &hACSR, queue );
    magma_cmfree(&hA, queue );
    magma_cmfree(&hL, queue );
    magma_cmfree(&hU, queue );

    return info;
}



/**
    Purpose
    -------

    Prepares the ILU transpose preconditioner via the cuSPARSE.

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                input matrix A

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/

extern "C" magma_int_t
magma_ccumilusetup_transpose(
    magma_c_matrix A,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_c_matrix Ah1={Magma_CSR}, Ah2={Magma_CSR};

    // transpose the matrix
    magma_cmtransfer( precond->L, &Ah1, Magma_DEV, Magma_CPU, queue );
    magma_cmconvert( Ah1, &Ah2, A.storage_type, Magma_CSR, queue );
    magma_cmfree(&Ah1, queue );
    magma_cmtransposeconjugate( Ah2, &Ah1, queue );
    magma_cmfree(&Ah2, queue );
    Ah2.blocksize = A.blocksize;
    Ah2.alignment = A.alignment;
    magma_cmconvert( Ah1, &Ah2, Magma_CSR, A.storage_type, queue );
    magma_cmfree(&Ah1, queue );
    magma_cmtransfer( Ah2, &(precond->LT), Magma_CPU, Magma_DEV, queue );
    magma_cmfree(&Ah2, queue );
    
    magma_cmtransfer( precond->U, &Ah1, Magma_DEV, Magma_CPU, queue );
    magma_cmconvert( Ah1, &Ah2, A.storage_type, Magma_CSR, queue );
    magma_cmfree(&Ah1, queue );
    magma_cmtransposeconjugate( Ah2, &Ah1, queue );
    magma_cmfree(&Ah2, queue );
    Ah2.blocksize = A.blocksize;
    Ah2.alignment = A.alignment;
    magma_cmconvert( Ah1, &Ah2, Magma_CSR, A.storage_type, queue );
    magma_cmfree(&Ah1, queue );
    magma_cmtransfer( Ah2, &(precond->UT), Magma_CPU, Magma_DEV, queue );
    magma_cmfree(&Ah2, queue );
   
    CHECK(magma_ctrisolve_analysis(precond->LT, &precond->cuinfoLT, true, false, false, queue));
    CHECK(magma_ctrisolve_analysis(precond->UT, &precond->cuinfoUT, false, false, false, queue));

cleanup:
    magma_cmfree(&Ah1, queue );
    magma_cmfree(&Ah2, queue );

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
    precond     magma_c_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/

extern "C" magma_int_t
magma_ccumilugeneratesolverinfo(
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_c_matrix hA={Magma_CSR}, hL={Magma_CSR}, hU={Magma_CSR};
    
    if (precond->L.memory_location != Magma_DEV ){
        CHECK( magma_cmtransfer( precond->M, &hA,
        precond->M.memory_location, Magma_CPU, queue ));

        hL.diagorder_type = Magma_UNITY;
        CHECK( magma_cmconvert( hA, &hL , Magma_CSR, Magma_CSRL, queue ));
        hU.diagorder_type = Magma_VALUE;
        CHECK( magma_cmconvert( hA, &hU , Magma_CSR, Magma_CSRU, queue ));
        CHECK( magma_cmtransfer( hL, &(precond->L), Magma_CPU, Magma_DEV, queue ));
        CHECK( magma_cmtransfer( hU, &(precond->U), Magma_CPU, Magma_DEV, queue ));
        
        magma_cmfree(&hA, queue );
        magma_cmfree(&hL, queue );
        magma_cmfree(&hU, queue );
    }
    
    CHECK(magma_ctrisolve_analysis(precond->L, &precond->cuinfoL, false, false, false, queue));
    CHECK(magma_ctrisolve_analysis(precond->U, &precond->cuinfoU, true, false, false, queue));
    
    if( precond->trisolver != 0 && precond->trisolver != Magma_CUSOLVE ){
        //prepare for iterative solves

        // extract the diagonal of L into precond->d
        CHECK( magma_cjacobisetup_diagscal( precond->L, &precond->d, queue ));
        CHECK( magma_cvinit( &precond->work1, Magma_DEV, precond->U.num_rows, 1, MAGMA_C_ZERO, queue ));
        
        // extract the diagonal of U into precond->d2
        CHECK( magma_cjacobisetup_diagscal( precond->U, &precond->d2, queue ));
        CHECK( magma_cvinit( &precond->work2, Magma_DEV, precond->U.num_rows, 1, MAGMA_C_ZERO, queue ));
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
    b           magma_c_matrix
                RHS

    @param[in,out]
    x           magma_c_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/

extern "C" magma_int_t
magma_capplycumilu_l(
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
        
    magmaFloatComplex one = MAGMA_C_MAKE( 1.0, 0.0);

    // CUSPARSE context //
    if( precond->trisolver == Magma_CUSOLVE || precond->trisolver == 0 ){
        CHECK(magma_ctrisolve(precond->L, precond->cuinfoL, false, false, false, b, *x, queue));
    } else if( precond->trisolver == Magma_SYNCFREESOLVE ){
        magma_cgecscsyncfreetrsm_solve( precond->L.num_rows,
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
    b           magma_c_matrix
                RHS

    @param[in,out]
    x           magma_c_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/
   
extern "C" magma_int_t
magma_capplycumilu_l_transpose(
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magmaFloatComplex one = MAGMA_C_MAKE( 1.0, 0.0);

    // CUSPARSE context //
    CHECK(magma_ctrisolve(precond->LT, precond->cuinfoLT, true, false, false, b, *x, queue));
    
    

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
    b           magma_c_matrix
                RHS

    @param[in,out]
    x           magma_c_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/

extern "C" magma_int_t
magma_capplycumilu_r(
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magmaFloatComplex one = MAGMA_C_MAKE( 1.0, 0.0);

    // CUSPARSE context //
    if( precond->trisolver == Magma_CUSOLVE || precond->trisolver == 0 ){
        CHECK(magma_ctrisolve(precond->U, precond->cuinfoU, true, false, false, b, *x, queue));
    } else if( precond->trisolver == Magma_SYNCFREESOLVE ){
        magma_cgecscsyncfreetrsm_solve( precond->U.num_rows,
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
    b           magma_c_matrix
                RHS

    @param[in,out]
    x           magma_c_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/

extern "C" magma_int_t
magma_capplycumilu_r_transpose(
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magmaFloatComplex one = MAGMA_C_MAKE( 1.0, 0.0);

    CHECK(magma_ctrisolve(precond->UT, precond->cuinfoUT, false, false, false, b, *x, queue));
    
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
    A           magma_c_matrix
                input matrix A

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_chepr
*******************************************************************************/

extern "C" magma_int_t
magma_ccumiccsetup(
    magma_c_matrix A,
    magma_c_preconditioner *precond,
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
    
    magma_c_matrix hA={Magma_CSR}, hACSR={Magma_CSR}, U={Magma_CSR};
    CHECK( magma_cmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
    U.diagorder_type = Magma_VALUE;
    CHECK( magma_cmconvert( hA, &hACSR, hA.storage_type, Magma_CSR, queue ));

    // in case using fill-in
    if( precond->levels > 0 ){
            magma_c_matrix hAL={Magma_CSR}, hAUt={Magma_CSR};
            CHECK( magma_csymbilu( &hACSR, precond->levels, &hAL, &hAUt,  queue ));
            magma_cmfree(&hAL, queue);
            magma_cmfree(&hAUt, queue);
    }

    CHECK( magma_cmconvert( hACSR, &U, Magma_CSR, Magma_CSRL, queue ));
    magma_cmfree( &hACSR, queue );
    CHECK( magma_cmtransfer(U, &(precond->M), Magma_CPU, Magma_DEV, queue ));

    // CUSPARSE context //
    CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
    CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrA ));
    cusparseCreateSolveAnalysisInfo( &(precond->cuinfoILU) );
    // use kernel to manually check for zeros n the diagonal
    CHECK( magma_cdiagcheck( precond->M, queue ) );
    
#if CUDA_VERSION >= 12000
    // this version has the bug fixed where a zero on the diagonal causes a crash
    CHECK_CUSPARSE( cusparseCreateCsric02Info(&info_M) );
    CHECK_CUSPARSE( cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_GENERAL ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrA, CUSPARSE_INDEX_BASE_ZERO ));
    int buffersize;
    int structural_zero;
    int numerical_zero;
    
    CHECK_CUSPARSE(
    cusparseCcsric02_bufferSize( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M,
                         &buffersize ) );
    
    CHECK( magma_malloc((void**)&pBuffer, buffersize) );

    CHECK_CUSPARSE( cusparseCcsric02_analysis( cusparseHandle,
            precond->M.num_rows, precond->M.nnz, descrA,
            precond->M.dval, precond->M.drow, precond->M.dcol,
            info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer ));
    CHECK_CUSPARSE( cusparseXcsric02_zeroPivot( cusparseHandle, info_M, &numerical_zero ) );
    CHECK_CUSPARSE( cusparseXcsric02_zeroPivot( cusparseHandle, info_M, &structural_zero ) );

    CHECK_CUSPARSE(
    cusparseCcsric02( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer) );    

#else
    // this version contains the bug but is needed for backward compability
    CHECK_CUSPARSE( cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrA, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrA, CUSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE( cusparseSetMatFillMode( descrA, CUSPARSE_FILL_MODE_LOWER ));
    
    // todo: Zcsric0 needs different analysis (cusparseCcsric02_analysis)
    cusparseCcsrsm_analysis( cusparseHandle,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             precond->M.num_rows, precond->M.nnz, descrA,
                             precond->M.dval, precond->M.drow, precond->M.dcol,
                             precond->cuinfoILU );
    cusparseCcsric0( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     precond->M.num_rows, precond->M.nnz, descrA,
                     precond->M.dval,
                     precond->M.drow,
                     precond->M.dcol,
                     precond->cuinfoILU );
#endif

    CHECK( magma_cmtransfer( precond->M, &precond->L, 
        Magma_DEV, Magma_DEV, queue ));
    CHECK( magma_cmtranspose(precond->M, &precond->U, queue ));

    if (precond->trisolver == 0 || precond->trisolver == Magma_CUSOLVE) {
        CHECK(magma_ccumicgeneratesolverinfo(precond, queue));
    } else {
        //prepare for iterative solves

        // extract the diagonal of L into precond->d
        CHECK(magma_cjacobisetup_diagscal(precond->L, &precond->d, queue));
        CHECK(magma_cvinit(&precond->work1, Magma_DEV, hA.num_rows, 1, 
            MAGMA_C_ZERO, queue));

        // extract the diagonal of U into precond->d2
        CHECK(magma_cjacobisetup_diagscal(precond->U, &precond->d2, queue));
        CHECK(magma_cvinit(&precond->work2, Magma_DEV, hA.num_rows, 1, 
            MAGMA_C_ZERO, queue));
    }



/*
    // to enable also the block-asynchronous iteration for the triangular solves
    CHECK( magma_cmtransfer( precond->M, &hA, Magma_DEV, Magma_CPU, queue ));
    hA.storage_type = Magma_CSR;

    magma_c_matrix hD, hR, hAt

    CHECK( magma_ccsrsplit( 256, hA, &hD, &hR, queue ));

    CHECK( magma_cmtransfer( hD, &precond->LD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_cmtransfer( hR, &precond->L, Magma_CPU, Magma_DEV, queue ));

    magma_cmfree(&hD, queue );
    magma_cmfree(&hR, queue );

    CHECK( magma_c_cucsrtranspose(   hA, &hAt, queue ));

    CHECK( magma_ccsrsplit( 256, hAt, &hD, &hR, queue ));

    CHECK( magma_cmtransfer( hD, &precond->UD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_cmtransfer( hR, &precond->U, Magma_CPU, Magma_DEV, queue ));
    
    magma_cmfree(&hD, queue );
    magma_cmfree(&hR, queue );
    magma_cmfree(&hA, queue );
    magma_cmfree(&hAt, queue );
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
    magma_cmfree(&U, queue );
    magma_cmfree(&hA, queue );

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
    precond     magma_c_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_chepr
    ********************************************************************/

extern "C" magma_int_t
magma_ccumicgeneratesolverinfo(
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    CHECK(magma_ctrisolve_analysis(precond->M, &precond->cuinfoL, false, false, false, queue));
    CHECK(magma_ctrisolve_analysis(precond->M, &precond->cuinfoU, false, false, true, queue));
    

/*
    // to enable also the block-asynchronous iteration for the triangular solves
    CHECK( magma_cmtransfer( precond->M, &hA, Magma_DEV, Magma_CPU, queue ));
    hA.storage_type = Magma_CSR;

    CHECK( magma_ccsrsplit( 256, hA, &hD, &hR, queue ));

    CHECK( magma_cmtransfer( hD, &precond->LD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_cmtransfer( hR, &precond->L, Magma_CPU, Magma_DEV, queue ));

    magma_cmfree(&hD, queue );
    magma_cmfree(&hR, queue );

    CHECK( magma_c_cucsrtranspose(   hA, &hAt, queue ));

    CHECK( magma_ccsrsplit( 256, hAt, &hD, &hR, queue ));

    CHECK( magma_cmtransfer( hD, &precond->UD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_cmtransfer( hR, &precond->U, Magma_CPU, Magma_DEV, queue ));
    
    magma_cmfree(&hD, queue );
    magma_cmfree(&hR, queue );
    magma_cmfree(&hA, queue );
    magma_cmfree(&hAt, queue );
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
    b           magma_c_matrix
                RHS

    @param[in,out]
    x           magma_c_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_chepr
    ********************************************************************/

extern "C" magma_int_t
magma_capplycumicc_l(
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magmaFloatComplex one = MAGMA_C_MAKE( 1.0, 0.0);

    CHECK(magma_ctrisolve(precond->M, precond->cuinfoL, false, false, false, b, *x, queue));

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
    b           magma_c_matrix
                RHS

    @param[in,out]
    x           magma_c_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_chepr
    ********************************************************************/

extern "C" magma_int_t
magma_capplycumicc_r(
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magmaFloatComplex one = MAGMA_C_MAKE( 1.0, 0.0);

    CHECK(magma_ctrisolve(precond->M, precond->cuinfoU, false, false, true, b, *x, queue));
    
    

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
    b           magma_c_matrix
                RHS

    @param[out]
    x           magma_c_matrix*
                vector to precondition

    @param[in]
    precond     magma_c_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_capplyiteric_l(
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t dofs = precond->L.num_rows;
    magma_c_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = precond->maxiter;

    // compute c = D^{-1}b and copy c as initial guess to x
    CHECK( magma_cjacobisetup_vector_gpu( dofs, b, precond->d,
                                                precond->work1, x, queue ));
    // Jacobi iterator
    CHECK( magma_cjacobiiter_precond( precond->L, x, &jacobiiter_par, precond , queue ));

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
    b           magma_c_matrix
                RHS

    @param[out]
    x           magma_c_matrix*
                vector to precondition

    @param[in]
    precond     magma_c_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_capplyiteric_r(
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_int_t dofs = precond->U.num_rows;
    magma_c_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = precond->maxiter;

    // compute c = D^{-1}b and copy c as initial guess to x
    CHECK( magma_cjacobisetup_vector_gpu( dofs, b, precond->d,
                                                precond->work1, x, queue ));

    // Jacobi iterator
    CHECK( magma_cjacobiiter_precond( precond->U, x, &jacobiiter_par, precond , queue ));
    
cleanup:
    return info;
}

