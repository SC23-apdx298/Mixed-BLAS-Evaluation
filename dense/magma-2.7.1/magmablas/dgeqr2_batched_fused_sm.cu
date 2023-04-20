/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @author Ahmad Abdelfattah
       @author Azzam Haidar

       @generated from magmablas/zgeqr2_batched_fused_sm.cu, normal z -> d, Wed Feb 22 15:21:26 2023
*/

#include <cuda.h>    // for CUDA_VERSION
#include "magma_internal.h"
#include "magma_templates.h"
#include "dgeqr2_batched_fused.cuh"
#include "batched_kernel_param.h"

#define PRECISION_d

////////////////////////////////////////////////////////////////////////////////
__global__
void
dgeqr2_fused_sm_kernel_batched(
    int M, int N,
    double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    double **dtau_array, magma_int_t taui,
    magma_int_t *info_array, magma_int_t batchCount)
{
    extern __shared__ double zdata[];
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int ntx = blockDim.x;
    const int nty = blockDim.y;
    const int batchid = blockIdx.x * nty + ty;
    if(batchid >= batchCount) return;

    const int slda  = SLDA(M);
    double* dA   = dA_array[batchid] + Aj * ldda + Ai;
    double* dtau = dtau_array[batchid] + taui;
    magma_int_t* info = &info_array[batchid];

    // shared memory pointers
    double* sA    = (double*)(zdata);
    double* sY    = sA   + (nty * slda * N);
    double* stau  = sY   + (nty * N);
    double* sTmp  = stau + nty * N;
    sA    += ty * slda * N;
    sY    += ty * N;
    stau  += ty * N;
    sTmp  += ty * ntx;
    double* snorm = (double*) (sTmp); // must be set after offsetting w.r.t. ty

    double alpha, tau, tmp, scale = MAGMA_D_ZERO;
    double norm = MAGMA_D_ZERO, beta;

    if( tx == 0 ){
        (*info) = 0;
    }

    // init tau
    if(tx < N) {
        stau[tx] = MAGMA_D_ZERO;
    }

    // read
    for(int j = 0; j < N; j++){
        for(int i = tx; i < M; i+=ntx) {
            sA(i,j) = dA[ j * ldda + i ];
        }
    }
    __syncthreads();

    for(int j = 0; j < N; j++){
        alpha = sA(j,j);

        dgeqr2_compute_norm(M-j, &sA(j,j), snorm, tx, ntx);
        // there is a sync at the end of dgeqr2_compute_norm

        norm = sqrt(snorm[0]);
        beta = -copysign(norm, real(alpha));
        scale = MAGMA_D_DIV( MAGMA_D_ONE,  alpha - MAGMA_D_MAKE(beta, 0));
        tau = MAGMA_D_MAKE( (beta - real(alpha)) / beta, -imag(alpha) / beta );

        if(tx == 0) {
            stau[j] = tau;
            sA(j,j) = MAGMA_D_ONE;
        }

        // scale the current column below the diagonal
        for(int i = (tx+j+1); i < M; i+=ntx) {
            sA(i,j) *= scale;
        }
        __syncthreads();

        // copy the first portion of the column into tmp
        // since M > N and ntx >= N, this portion must
        // have the diagonal
        tmp = (tx == j) ? MAGMA_D_MAKE(beta, MAGMA_D_ZERO) : sA(tx, j);

        // write the column into global memory
        dA[j * ldda + tx] = tmp;
        for(int i = tx+ntx; i < M; i+=ntx) {
            dA[ j * ldda + i ] = sA(i, j);
        }

        // now compute (I - tau * v * v') A
        // first: y = tau * v' * A (row vector)
        dgeqr2_compute_vtA_device(M, N, j, sA, slda, sY, tau, sTmp, tx, ntx);
        __syncthreads();

        // now compute: A = A - v * y
        for(int jj = j+1; jj < N; jj++){
            for(int i = tx+j; i < M; i+=ntx) {
                sA(i,jj) -= sA(i,j) * sY[jj];
            }
        }
        __syncthreads();
    }

    // write tau and the last column
    if(tx < N) {
        dtau[tx] = stau[tx];
    }
}

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_dgeqr2_fused_sm_batched(
    magma_int_t m, magma_int_t n,
    double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    double **dtau_array, magma_int_t taui,
    magma_int_t* info_array, magma_int_t nthreads, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_device_t device;
    magma_getdevice( &device );

    if (m < 0)
        arginfo = -1;
    else if (n < 0)
        arginfo = -2;
    else if (ldda < max(1,m))
        arginfo = -4;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return arginfo;

    // disable this kernel for n > 8
    if( m < n || n > 8) return -100;

    nthreads = min(nthreads, m);

    const magma_int_t ntcol = 1;
    magma_int_t shmem = ( SLDA(m) * n * sizeof(double) );
    shmem            += ( n        * sizeof(double) );  // sY
    shmem            += ( n        * sizeof(double) );  // stau
    shmem            += ( nthreads * sizeof(double) );  // used for snorm and for computing v' * A
    shmem            *= ntcol;
    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 grid(gridx, 1, 1);
    dim3 threads( nthreads, ntcol, 1);

    // get max. dynamic shared memory on the GPU
    int nthreads_max, shmem_max = 0;
    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(dgeqr2_fused_sm_kernel_batched, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        // printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
        arginfo = -100;
        return arginfo;
    }

    if( check_launch_only == 1 ) return arginfo;

    void *kernel_args[] = {&m, &n, &dA_array, &Ai, &Aj, &ldda, &dtau_array, &taui, &info_array, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)dgeqr2_fused_sm_kernel_batched, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        // printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}
