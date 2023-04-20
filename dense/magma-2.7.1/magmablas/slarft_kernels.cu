/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @generated from magmablas/zlarft_kernels.cu, normal z -> s, Wed Feb 22 15:21:16 2023
       @author Azzam Haidar
*/

#include "magma_internal.h"
#include "magma_templates.h"

#define use_gemm_larft

/******************************************************************************/
static  __device__ void
slarft_strmv_sm32x32_device(
    int n, int k, float *tau,
    float *Tin, int ldtin,  float *Tout, int ldtout )
{
    extern __shared__ float shared_data[];

    int tx = threadIdx.x;
    float *sdata = (float*)shared_data;
    float res;

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
        res = MAGMA_S_ZERO;
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
slarft_strmv_sm32x32_kernel(
    int n, int k, float *tau,
    float *Tin, int ldtin,  float *Tout, int ldtout )
{
    slarft_strmv_sm32x32_device( n, k, tau, Tin, ldtin, Tout, ldtout);
}


/******************************************************************************/
__global__ void
slarft_strmv_sm32x32_kernel_batched(
    int n, int k,
    float **tau_array, int taui,
    float **Tin_array,  int Tini, int Tinj, int ldtin,
    float **Tout_array, int Touti, int Toutj, int ldtout )
{
    int batchId = blockIdx.z;
    float *tau  = tau_array[batchId]  + taui;
    float *Tin  = Tin_array[batchId]  + Tinj  * ldtin + Tini;
    float *Tout = Tout_array[batchId] + Toutj * ldtout + Touti;
    slarft_strmv_sm32x32_device( n, k, tau, Tin, ldtin, Tout, ldtout);
}


/******************************************************************************/
extern "C"
void magmablas_slarft_strmv_sm32x32(
    magma_int_t m, magma_int_t n,
    float *tau,
    float *Tin, magma_int_t ldtin,
    float *Tout, magma_int_t ldtout,
    magma_queue_t queue )
{
    dim3 grid(1);
    dim3 threads(max(m,1), 1, 1);
    size_t shmem = sizeof(float)*(m*m);
    slarft_strmv_sm32x32_kernel
        <<< grid, threads, shmem, queue->cuda_stream() >>>
        (m, n,  tau, Tin, ldtin, Tout, ldtout);
}


/******************************************************************************/
extern "C"
void magmablas_slarft_strmv_sm32x32_batched(
    magma_int_t m, magma_int_t n,
    float **tau_array, magma_int_t taui,
    float **Tin_array, magma_int_t Tini, magma_int_t Tinj, magma_int_t ldtin,
    float **Tout_array, magma_int_t Touti, magma_int_t Toutj, magma_int_t ldtout,
    magma_int_t batchCount, magma_queue_t queue)
{

    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(max(m,1), 1, 1);
    size_t shmem = sizeof(float)*(m*m);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid(1, 1, ibatch);

        slarft_strmv_sm32x32_kernel_batched
        <<< grid, threads, shmem, queue->cuda_stream() >>>
        (m, n,  tau_array+i, taui,
         Tin_array+i,  Tini,  Tinj,  ldtin,
         Tout_array+i, Touti, Toutj, ldtout);
    }
}


/******************************************************************************/
static __device__ void
slarft_recstrmv_sm32x32_device(
    int m, int n, float *tau,
    float *Trec, int ldtrec, float *Ttri, int ldttri)
{
    extern __shared__ float shared_data[];

    int tx = threadIdx.x;
    float *sdata = (float*)shared_data;
    float res;

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
        res = MAGMA_S_ZERO;
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
slarft_recstrmv_sm32x32_kernel(
    int m, int n, float *tau,
    float *Trec, int ldtrec, float *Ttri, int ldttri)
{
    slarft_recstrmv_sm32x32_device(m, n, tau, Trec, ldtrec, Ttri, ldttri);
}


/******************************************************************************/
__global__ void
slarft_recstrmv_sm32x32_kernel_batched(
    int m, int n,
    float **tau_array, int taui,
    float **Trec_array, int Treci, int Trecj, int ldtrec,
    float **Ttri_array, int Ttrii, int Ttrij, int ldttri)
{
    int batchId = blockIdx.z;
    float *tau  = tau_array[batchId]  + taui;
    float *Trec = Trec_array[batchId] + Trecj * ldtrec + Treci;
    float *Ttri = Ttri_array[batchId] + Ttrij * ldttri + Ttrii;
    slarft_recstrmv_sm32x32_device(m, n, tau, Trec, ldtrec, Ttri, ldttri);
}


/******************************************************************************/
extern "C"
void magmablas_slarft_recstrmv_sm32x32(
    magma_int_t m, magma_int_t n,
    float *tau,
    float *Trec, magma_int_t ldtrec,
    float *Ttri, magma_int_t ldttri,
    magma_queue_t queue )
{
    dim3 grid(1);
    dim3 threads(max(m,1), 1, 1);
    size_t shmem = sizeof(float)*(m*n);
    slarft_recstrmv_sm32x32_kernel
        <<< grid, threads, shmem, queue->cuda_stream() >>>
        (m, n,  tau, Trec, ldtrec, Ttri, ldttri);
}


/******************************************************************************/
extern "C"
void magmablas_slarft_recstrmv_sm32x32_batched(
    magma_int_t m, magma_int_t n,
    float **tau_array, magma_int_t taui,
    float **Trec_array, magma_int_t Treci, magma_int_t Trecj, magma_int_t ldtrec,
    float **Ttri_array, magma_int_t Ttrii, magma_int_t Ttrij, magma_int_t ldttri,
    magma_int_t batchCount, magma_queue_t queue)
{
    dim3 threads(max(m,1), 1, 1);
    magma_int_t max_batchCount = queue->get_maxBatch();
    size_t shmem = sizeof(float)*(m*n);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid(1, 1, ibatch);

        slarft_recstrmv_sm32x32_kernel_batched
        <<< grid, threads, shmem, queue->cuda_stream() >>>
        ( m, n,  tau_array+i, taui,
          Trec_array+i, Treci, Trecj, ldtrec,
          Ttri_array+i, Ttrii, Ttrij, ldttri);

    }
}
