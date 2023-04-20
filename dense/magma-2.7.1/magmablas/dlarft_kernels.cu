/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @generated from magmablas/zlarft_kernels.cu, normal z -> d, Wed Feb 22 15:21:16 2023
       @author Azzam Haidar
*/

#include "magma_internal.h"
#include "magma_templates.h"

#define use_gemm_larft

/******************************************************************************/
static  __device__ void
dlarft_dtrmv_sm32x32_device(
    int n, int k, double *tau,
    double *Tin, int ldtin,  double *Tout, int ldtout )
{
    extern __shared__ double shared_data[];

    int tx = threadIdx.x;
    double *sdata = (double*)shared_data;
    double res;

    // this routine apply a sequence of trmv to update k column of the triangular
    // T starting at n-k to n where T is of size n by n and where the first n-k
    // columns of T are supposed updated previously.
    // So the routine load all of T nxn to the shared memory
    // and apply the sequence of trmv.
    // to update a certain column i, threads go in horizontal fashion where
    // every thread read one row and do it gemv(dot) to generate
    // one element of the column of T then move to the next column

    // read T into shared
    for (int s=0; s < n-k; s++)
    {
        sdata[tx + s*n] = Tin[tx + s * ldtin];
    }

#if defined(use_gemm_larft)
    for (int s=n-k; s < n; s++)
    {
        if (tx == s)
            sdata[tx + s*n] = tau[s];
        else
            sdata[tx + s*n] = -tau[s] * Tin[tx + s * ldtin];
    }
#else
    for (int s=n-k; s < n; s++)
    {
        sdata[tx + s*n] = Tin[tx + s * ldtin];
    }
#endif

    // perform trmv
    for (int i=n-k; i < n; i++)
    {
        __syncthreads();
        res = MAGMA_D_ZERO;
        if (tx < i)
        {
            for (int j=tx; j < i; j++)
            {
                res += sdata[tx + j * n] * sdata[j+ i * n];
            }
        }
        __syncthreads();
        if (tx < i)
        {
            sdata[tx + i * n] = res;
        }
    }

    __syncthreads();
    // write back the updated block of k column of T
    for (int s=n-k; s < n; s++)
    {
        Tout[tx + s * ldtout] = sdata[tx + s*n];
    }
}


/******************************************************************************/
__global__ void
dlarft_dtrmv_sm32x32_kernel(
    int n, int k, double *tau,
    double *Tin, int ldtin,  double *Tout, int ldtout )
{
    dlarft_dtrmv_sm32x32_device( n, k, tau, Tin, ldtin, Tout, ldtout);
}


/******************************************************************************/
__global__ void
dlarft_dtrmv_sm32x32_kernel_batched(
    int n, int k,
    double **tau_array, int taui,
    double **Tin_array,  int Tini, int Tinj, int ldtin,
    double **Tout_array, int Touti, int Toutj, int ldtout )
{
    int batchId = blockIdx.z;
    double *tau  = tau_array[batchId]  + taui;
    double *Tin  = Tin_array[batchId]  + Tinj  * ldtin + Tini;
    double *Tout = Tout_array[batchId] + Toutj * ldtout + Touti;
    dlarft_dtrmv_sm32x32_device( n, k, tau, Tin, ldtin, Tout, ldtout);
}


/******************************************************************************/
extern "C"
void magmablas_dlarft_dtrmv_sm32x32(
    magma_int_t m, magma_int_t n,
    double *tau,
    double *Tin, magma_int_t ldtin,
    double *Tout, magma_int_t ldtout,
    magma_queue_t queue )
{
    dim3 grid(1);
    dim3 threads(max(m,1), 1, 1);
    size_t shmem = sizeof(double)*(m*m);
    dlarft_dtrmv_sm32x32_kernel
        <<< grid, threads, shmem, queue->cuda_stream() >>>
        (m, n,  tau, Tin, ldtin, Tout, ldtout);
}


/******************************************************************************/
extern "C"
void magmablas_dlarft_dtrmv_sm32x32_batched(
    magma_int_t m, magma_int_t n,
    double **tau_array, magma_int_t taui,
    double **Tin_array, magma_int_t Tini, magma_int_t Tinj, magma_int_t ldtin,
    double **Tout_array, magma_int_t Touti, magma_int_t Toutj, magma_int_t ldtout,
    magma_int_t batchCount, magma_queue_t queue)
{

    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(max(m,1), 1, 1);
    size_t shmem = sizeof(double)*(m*m);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid(1, 1, ibatch);

        dlarft_dtrmv_sm32x32_kernel_batched
        <<< grid, threads, shmem, queue->cuda_stream() >>>
        (m, n,  tau_array+i, taui,
         Tin_array+i,  Tini,  Tinj,  ldtin,
         Tout_array+i, Touti, Toutj, ldtout);
    }
}


/******************************************************************************/
static __device__ void
dlarft_recdtrmv_sm32x32_device(
    int m, int n, double *tau,
    double *Trec, int ldtrec, double *Ttri, int ldttri)
{
    extern __shared__ double shared_data[];

    int tx = threadIdx.x;
    double *sdata = (double*)shared_data;
    double res;

    // to update a certain column i, threads go in horizontal fashion where
    // every thread read one row and do it gemv(dot) to generate
    // one element of the column of T then move to the next column

    // read T into shared
    for (int s=0; s < n; s++)
    {
        sdata[tx + s*n] = Trec[tx + s * ldtrec];
    }
    __syncthreads();

    // perform sequence of n-1 gemv
    for (int i=0; i < n; i++)
    {
        res = MAGMA_D_ZERO;
        for (int j=0; j < i; j++)
        {
            res += sdata[tx + j * n] * Ttri[j+ i * ldttri];
        }
        __syncthreads();   // a enlever
        sdata[tx + i * n] = -tau[i] * (sdata[tx + i * n] + res);
        __syncthreads();
    }

    // write back the updated block of k column of T  multiplying by -tau
    for (int s=0; s < n; s++)
    {
        Trec[tx + s * ldtrec] = sdata[tx + s*n];
    }
}


/******************************************************************************/
__global__ void
dlarft_recdtrmv_sm32x32_kernel(
    int m, int n, double *tau,
    double *Trec, int ldtrec, double *Ttri, int ldttri)
{
    dlarft_recdtrmv_sm32x32_device(m, n, tau, Trec, ldtrec, Ttri, ldttri);
}


/******************************************************************************/
__global__ void
dlarft_recdtrmv_sm32x32_kernel_batched(
    int m, int n,
    double **tau_array, int taui,
    double **Trec_array, int Treci, int Trecj, int ldtrec,
    double **Ttri_array, int Ttrii, int Ttrij, int ldttri)
{
    int batchId = blockIdx.z;
    double *tau  = tau_array[batchId]  + taui;
    double *Trec = Trec_array[batchId] + Trecj * ldtrec + Treci;
    double *Ttri = Ttri_array[batchId] + Ttrij * ldttri + Ttrii;
    dlarft_recdtrmv_sm32x32_device(m, n, tau, Trec, ldtrec, Ttri, ldttri);
}


/******************************************************************************/
extern "C"
void magmablas_dlarft_recdtrmv_sm32x32(
    magma_int_t m, magma_int_t n,
    double *tau,
    double *Trec, magma_int_t ldtrec,
    double *Ttri, magma_int_t ldttri,
    magma_queue_t queue )
{
    dim3 grid(1);
    dim3 threads(max(m,1), 1, 1);
    size_t shmem = sizeof(double)*(m*n);
    dlarft_recdtrmv_sm32x32_kernel
        <<< grid, threads, shmem, queue->cuda_stream() >>>
        (m, n,  tau, Trec, ldtrec, Ttri, ldttri);
}


/******************************************************************************/
extern "C"
void magmablas_dlarft_recdtrmv_sm32x32_batched(
    magma_int_t m, magma_int_t n,
    double **tau_array, magma_int_t taui,
    double **Trec_array, magma_int_t Treci, magma_int_t Trecj, magma_int_t ldtrec,
    double **Ttri_array, magma_int_t Ttrii, magma_int_t Ttrij, magma_int_t ldttri,
    magma_int_t batchCount, magma_queue_t queue)
{
    dim3 threads(max(m,1), 1, 1);
    magma_int_t max_batchCount = queue->get_maxBatch();
    size_t shmem = sizeof(double)*(m*n);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid(1, 1, ibatch);

        dlarft_recdtrmv_sm32x32_kernel_batched
        <<< grid, threads, shmem, queue->cuda_stream() >>>
        ( m, n,  tau_array+i, taui,
          Trec_array+i, Treci, Trecj, ldtrec,
          Ttri_array+i, Ttrii, Ttrij, ldttri);

    }
}
