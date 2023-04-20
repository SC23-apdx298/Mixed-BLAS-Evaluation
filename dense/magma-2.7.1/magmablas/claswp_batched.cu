/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @generated from magmablas/zlaswp_batched.cu, normal z -> c, Wed Feb 22 15:21:28 2023

       @author Azzam Haidar
       @author Tingxing Dong
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "claswp_device.cuh"

/******************************************************************************/
// parallel swap the swaped dA(1:nb,i:n) is stored in dout
__global__
void claswp_rowparallel_kernel(
                                int n, int width, int height,
                                magmaFloatComplex *dinput, int ldi,
                                magmaFloatComplex *doutput, int ldo,
                                magma_int_t*  pivinfo)
{
    claswp_rowparallel_devfunc(n, width, height, dinput, ldi, doutput, ldo, pivinfo);
}


/******************************************************************************/
__global__
void claswp_rowparallel_kernel_batched(
                                int n, int width, int height,
                                magmaFloatComplex **input_array, int input_i, int input_j, int ldi,
                                magmaFloatComplex **output_array, int output_i, int output_j, int ldo,
                                magma_int_t** pivinfo_array)
{
    int batchid = blockIdx.z;
    claswp_rowparallel_devfunc( n, width, height,
                                input_array[batchid]  + input_j  * ldi +  input_i, ldi,
                                output_array[batchid] + output_j * ldo + output_i, ldo,
                                pivinfo_array[batchid]);
}


/******************************************************************************/
extern "C" void
magma_claswp_rowparallel_batched( magma_int_t n,
                       magmaFloatComplex**  input_array, magma_int_t  input_i, magma_int_t  input_j, magma_int_t ldi,
                       magmaFloatComplex** output_array, magma_int_t output_i, magma_int_t output_j, magma_int_t ldo,
                       magma_int_t k1, magma_int_t k2,
                       magma_int_t **pivinfo_array,
                       magma_int_t batchCount, magma_queue_t queue)
{
#define  input_array(i,j)  input_array, i, j
#define output_array(i,j) output_array, i, j

    if (n == 0 ) return;
    int height = k2-k1;
    if ( height  > 1024)
    {
        fprintf( stderr, "%s: n=%lld > 1024, not supported\n", __func__, (long long) n );
    }

    int blocks = magma_ceildiv( n, SWP_WIDTH );
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3  grid(blocks, 1, ibatch);

        if ( n < SWP_WIDTH) {
            size_t shmem = sizeof(magmaFloatComplex) * height * n;
            claswp_rowparallel_kernel_batched
            <<< grid, height, shmem, queue->cuda_stream() >>>
            ( n, n, height, input_array+i, input_i, input_j, ldi, output_array+i, output_i, output_j, ldo, pivinfo_array+i );
        }
        else {
            size_t shmem = sizeof(magmaFloatComplex) * height * SWP_WIDTH;
            claswp_rowparallel_kernel_batched
            <<< grid, height, shmem, queue->cuda_stream() >>>
            ( n, SWP_WIDTH, height, input_array+i, input_i, input_j, ldi, output_array+i, output_i, output_j, ldo, pivinfo_array+i );
        }
    }
#undef  input_array
#undef output_attay
}


/******************************************************************************/
extern "C" void
magma_claswp_rowparallel_native(
    magma_int_t n,
    magmaFloatComplex* input, magma_int_t ldi,
    magmaFloatComplex* output, magma_int_t ldo,
    magma_int_t k1, magma_int_t k2,
    magma_int_t *pivinfo,
    magma_queue_t queue)
{
    if (n == 0 ) return;
    int height = k2-k1;
    if ( height  > MAX_NTHREADS)
    {
        fprintf( stderr, "%s: height=%lld > %lld, magma_claswp_rowparallel_q not supported\n",
                 __func__, (long long) n, (long long) MAX_NTHREADS );
    }

    int blocks = magma_ceildiv( n, SWP_WIDTH );
    dim3  grid(blocks, 1, 1);

    if ( n < SWP_WIDTH)
    {
        size_t shmem = sizeof(magmaFloatComplex) * height * n;
        claswp_rowparallel_kernel
            <<< grid, height, shmem, queue->cuda_stream() >>>
            ( n, n, height, input, ldi, output, ldo, pivinfo );
    }
    else
    {
        size_t shmem = sizeof(magmaFloatComplex) * height * SWP_WIDTH;
        claswp_rowparallel_kernel
            <<< grid, height, shmem, queue->cuda_stream() >>>
            ( n, SWP_WIDTH, height, input, ldi, output, ldo, pivinfo );
    }
}


/******************************************************************************/
// serial swap that does swapping one row by one row
__global__ void claswp_rowserial_kernel_batched( int n, magmaFloatComplex **dA_array, int lda, int k1, int k2, magma_int_t** ipiv_array )
{
    magmaFloatComplex* dA = dA_array[blockIdx.z];
    magma_int_t *dipiv = ipiv_array[blockIdx.z];

    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;

    k1--;
    k2--;

    if (tid < n) {
        magmaFloatComplex A1;

        for (int i1 = k1; i1 < k2; i1++)
        {
            int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
            if ( i2 != i1)
            {
                A1 = dA[i1 + tid * lda];
                dA[i1 + tid * lda] = dA[i2 + tid * lda];
                dA[i2 + tid * lda] = A1;
            }
        }
    }
}


/******************************************************************************/
// serial swap that does swapping one row by one row
__global__ void claswp_rowserial_kernel_native( int n, magmaFloatComplex_ptr dA, int lda, int k1, int k2, magma_int_t* dipiv )
{
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;

    //k1--;
    //k2--;

    if (tid < n) {
        magmaFloatComplex A1;

        for (int i1 = k1; i1 < k2; i1++)
        {
            int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
            if ( i2 != i1)
            {
                A1 = dA[i1 + tid * lda];
                dA[i1 + tid * lda] = dA[i2 + tid * lda];
                dA[i2 + tid * lda] = A1;
            }
        }
    }
}


/******************************************************************************/
// serial swap that does swapping one row by one row, similar to LAPACK
// K1, K2 are in Fortran indexing
extern "C" void
magma_claswp_rowserial_batched(magma_int_t n, magmaFloatComplex** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array,
                   magma_int_t batchCount, magma_queue_t queue)
{
    if (n == 0) return;

    int blocks = magma_ceildiv( n, BLK_SIZE );
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3  grid(blocks, 1, ibatch);

        magma_int_t max_BLK_SIZE__n = max(BLK_SIZE, n);
        claswp_rowserial_kernel_batched
        <<< grid, max_BLK_SIZE__n, 0, queue->cuda_stream() >>>
        (n, dA_array+i, lda, k1, k2, ipiv_array+i);
    }
}



/******************************************************************************/
// serial swap that does swapping one row by one row, similar to LAPACK
// K1, K2 are in Fortran indexing
extern "C" void
magma_claswp_rowserial_native(magma_int_t n, magmaFloatComplex_ptr dA, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t* dipiv, magma_queue_t queue)
{
    if (n == 0) return;

    int blocks = magma_ceildiv( n, BLK_SIZE );
    dim3  grid(blocks, 1, 1);

    size_t max_BLK_SIZE_n = max(BLK_SIZE, n);
    claswp_rowserial_kernel_native
        <<< grid, max_BLK_SIZE_n, 0, queue->cuda_stream() >>>
        (n, dA, lda, k1, k2, dipiv);
}



/******************************************************************************/
// serial swap that does swapping one column by one column
__device__ void claswp_columnserial_devfunc(int n, magmaFloatComplex_ptr dA, int lda, int k1, int k2, magma_int_t* dipiv )
{
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    k1--;
    k2--;
    if ( k1 < 0 || k2 < 0 ) return;


    if ( tid < n) {
        magmaFloatComplex A1;
        if (k1 <= k2)
        {
            for (int i1 = k1; i1 <= k2; i1++)
            {
                int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
                if ( i2 != i1)
                {
                    A1 = dA[i1 * lda + tid];
                    dA[i1 * lda + tid] = dA[i2 * lda + tid];
                    dA[i2 * lda + tid] = A1;
                }
            }
        } else
        {

            for (int i1 = k1; i1 >= k2; i1--)
            {
                int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
                if ( i2 != i1)
                {
                    A1 = dA[i1 * lda + tid];
                    dA[i1 * lda + tid] = dA[i2 * lda + tid];
                    dA[i2 * lda + tid] = A1;
                }
            }
        }
    }
}


__global__ void claswp_columnserial_kernel_batched( int n, magmaFloatComplex **dA_array, int lda, int k1, int k2, magma_int_t** ipiv_array )
{
    magmaFloatComplex* dA = dA_array[blockIdx.z];
    magma_int_t *dipiv = ipiv_array[blockIdx.z];

    claswp_columnserial_devfunc(n, dA, lda, k1, k2, dipiv);
}

__global__ void claswp_columnserial_kernel( int n, magmaFloatComplex_ptr dA, int lda, int k1, int k2, magma_int_t* dipiv )
{
    claswp_columnserial_devfunc(n, dA, lda, k1, k2, dipiv);
}

/******************************************************************************/
// serial swap that does swapping one column by one column
// K1, K2 are in Fortran indexing
extern "C" void
magma_claswp_columnserial(
    magma_int_t n, magmaFloatComplex_ptr dA, magma_int_t lda,
    magma_int_t k1, magma_int_t k2,
    magma_int_t *dipiv, magma_queue_t queue)
{
    if (n == 0 ) return;

    int blocks = magma_ceildiv( n, CLASWP_COL_NTH );
    dim3  grid(blocks, 1, 1);

    claswp_columnserial_kernel<<< grid, CLASWP_COL_NTH, 0, queue->cuda_stream() >>>
    (n, dA, lda, k1, k2, dipiv);
}

extern "C" void
magma_claswp_columnserial_batched(magma_int_t n, magmaFloatComplex** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array,
                   magma_int_t batchCount, magma_queue_t queue)
{
    if (n == 0 ) return;

    int blocks = magma_ceildiv( n, CLASWP_COL_NTH );

    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3  grid(blocks, 1, ibatch);

        magma_int_t min_CLASWP_COL_NTH__n = min(CLASWP_COL_NTH, n);
        claswp_columnserial_kernel_batched
        <<< grid, min_CLASWP_COL_NTH__n, 0, queue->cuda_stream() >>>
        (n, dA_array+i, lda, k1, k2, ipiv_array+i);
    }
}
