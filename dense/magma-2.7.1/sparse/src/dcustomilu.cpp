/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @author Hartwig Anzt

       @generated from sparse/src/zcustomilu.cpp, normal z -> d, Wed Feb 22 15:21:58 2023
*/
#include "magmasparse_internal.h"


#include "../blas/magma_trisolve.h"

#define REAL

/* For hipSPARSE, they use a separate real type than for hipBLAS */
#ifdef MAGMA_HAVE_HIP
  #define double double
#endif

/**
    Purpose
    -------

    Reads in an Incomplete LU preconditioner.

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input matrix A
                
    @param[in]
    b           magma_d_matrix
                input RHS b

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_dcustomilusetup(
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_d_matrix hA={Magma_CSR};
    char preconditionermatrix[255];
    
    // first L
    snprintf( preconditionermatrix, sizeof(preconditionermatrix),
                "/Users/hanzt0114cl306/work/matrices/matrices/ILUT_L.mtx" );
    
    CHECK( magma_d_csr_mtx( &hA, preconditionermatrix , queue) );
    CHECK( magma_dmtransfer( hA, &precond->L, Magma_CPU, Magma_DEV , queue ));
    // extract the diagonal of L into precond->d
    CHECK( magma_djacobisetup_diagscal( precond->L, &precond->d, queue ));
    CHECK( magma_dvinit( &precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_D_ZERO, queue ));

    magma_dmfree( &hA, queue );
    
    // now U
    snprintf( preconditionermatrix, sizeof(preconditionermatrix),
                "/Users/hanzt0114cl306/work/matrices/matrices/ILUT_U.mtx" );

    CHECK( magma_d_csr_mtx( &hA, preconditionermatrix , queue) );
    CHECK( magma_dmtransfer( hA, &precond->U, Magma_CPU, Magma_DEV , queue ));
    // extract the diagonal of U into precond->d2
    CHECK( magma_djacobisetup_diagscal( precond->U, &precond->d2, queue ));
    CHECK( magma_dvinit( &precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_D_ZERO, queue ));

    CHECK(magma_dtrisolve_analysis(precond->L, &precond->cuinfoL, false, true, false, queue));
    CHECK(magma_dtrisolve_analysis(precond->U, &precond->cuinfoU, true, false, false, queue));
        
cleanup:        
    magma_dmfree( &hA, queue );
    
    return info;
}
    
