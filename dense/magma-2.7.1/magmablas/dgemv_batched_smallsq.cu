/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @author Azzam Haidar

       @generated from magmablas/zgemv_batched_smallsq.cu, normal z -> d, Wed Feb 22 15:21:25 2023
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"

#define PRECISION_d

#define SLDA(N)              ( (N==15||N==23||N==31)? N : (N+1) )

#ifdef MAGMA_HAVE_CUDA
#define MAX_THREADS          (128)
#else
#define MAX_THREADS          (256)
#endif
#define BATCH_GEMV_NTCOL(N)  (max(1,MAX_THREADS/N))

////////////////////////////////////////////////////////////////////////////////
template<typename T, int N>
__global__ __launch_bounds__(N*BATCH_GEMV_NTCOL(N)) void
dgemvn_batched_smallsq_kernel(
        const T alpha,
        T const * const * dA_array, const T* dA, int ldda, int strideA,
        T const * const * dx_array, const T* dx, int incx, int stridex,
        const T beta,
        T**       dy_array,       T* dy, int incy, int stridey,
        const int batchCount)
{
    extern __shared__ T zdata[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;

    const int batchid = bx * blockDim.y + ty;
    if(batchid >= batchCount) return;

    const T* A = (dA_array == NULL) ? dA + batchid * strideA : dA_array[batchid];
    const T* x = (dx_array == NULL) ? dx + batchid * stridex : dx_array[batchid];
    T* y = (dy_array == NULL) ? dy + batchid * stridey : dy_array[batchid];

    T rA[N] = {MAGMA_D_ZERO};

    // shared memory
    T* sx = (T*)(zdata);
    sx += ty * N;

    // read x in shmem
    sx[tx] = x[tx * incx];
    __syncthreads();

    T ry = (beta == MAGMA_D_ZERO) ? MAGMA_D_ZERO : beta * y[tx * incy];
    #pragma unroll
    for(int j = 0; j < N; j++) {
        rA[j] = A[j * ldda + tx];
    }

    T rTmp = MAGMA_D_ZERO;
    #pragma unroll
    for(int j = 0; j < N; j++) {
        rTmp += rA[j] * sx[j];
    }

    rTmp *= alpha;
    ry   += rTmp;
    y[tx * incy] = ry;

}

////////////////////////////////////////////////////////////////////////////////
template<typename T, int N>
__global__ __launch_bounds__(N*BATCH_GEMV_NTCOL(N)) void
dgemvc_batched_smallsq_kernel(
        const magma_trans_t transA, const T alpha,
        T const * const * dA_array, const T* dA, int ldda, int strideA,
        T const * const * dx_array, const T* dx, int incx, int stridex,
        const T beta,
        T**      dy_array,       T* dy, int incy, int stridey,
        const int batchCount)
{
    extern __shared__ T zdata[];

    const int tx   = threadIdx.x;
    const int ty   = threadIdx.y;
    const int bx   = blockIdx.x;
    const int nty  = blockDim.y;
    const int slda = SLDA(N);

    const int batchid = bx * blockDim.y + ty;
    if(batchid >= batchCount) return;

    const T* A = (dA_array == NULL) ? dA + batchid * strideA : dA_array[batchid];
    const T* x = (dx_array == NULL) ? dx + batchid * stridex : dx_array[batchid];
    T* y = (dy_array == NULL) ? dy + batchid * stridey : dy_array[batchid];

    T rA[N] = {MAGMA_D_ZERO};

    // shared memory
    T* sA = (T*)(zdata);
    T* sx = sA + nty * slda * N;
    sA += ty * slda * N;
    sx += ty * N;

    T ry = (beta == MAGMA_D_ZERO) ? MAGMA_D_ZERO : beta * y[tx * incy];

    // read x in shmem
    sx[tx] = x[tx * incx];

    #pragma unroll
    for(int j = 0; j < N; j++) {
        rA[j] = A[j * ldda + tx];
    }

    // transpose
    #pragma unroll
    for(int j = 0; j < N; j++) {
        #if defined(PRECISION_z) || defined(PRECISION_c)
        sA[tx * slda + j] = (transA == MagmaConjTrans) ? MAGMA_D_CONJ(rA[j]) : rA[j];
        #else
        sA[tx * slda + j] = rA[j];
        #endif
    }
    __syncthreads();

    #pragma unroll
    for(int j = 0; j < N; j++) {
         rA[j] = sA[j * slda + tx];
    }

    T rTmp = MAGMA_D_ZERO;
    #pragma unroll
    for(int j = 0; j < N; j++) {
        rTmp += rA[j] * sx[j];
    }

    rTmp *= alpha;
    ry   += rTmp;
    y[tx * incy] = ry;
}

////////////////////////////////////////////////////////////////////////////////
template<int N>
static int
dgemv_batched_smallsq_kernel_driver(
    magma_trans_t transA,
    const double alpha,
    double const * const * dA_array, const double* dA, magma_int_t ldda, magma_int_t strideA,
    double const * const * dx_array, const double* dx, magma_int_t incx, magma_int_t stridex,
    const double beta,
    double**      dy_array,       double* dy, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t ntcol  = BATCH_GEMV_NTCOL(N);
    magma_int_t shmem  = N * sizeof(double);
    if( !(transA == MagmaNoTrans) ) {
        const int slda = SLDA(N);
        shmem += slda * N * sizeof(double);
    }
    shmem *= ntcol;

    // get max. dynamic shared memory on the GPU
    int shmem_max, nthreads_max;
    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(dgemvn_batched_smallsq_kernel<double, N>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
        cudaFuncSetAttribute(dgemvc_batched_smallsq_kernel<double, N>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = N * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        return -100;
    }

    const int nblocks = magma_ceildiv(batchCount, ntcol);
    dim3 grid(nblocks, 1, 1);
    dim3 threads(N, ntcol, 1);
    if( transA == MagmaNoTrans ) {
        void *kernel_args[] = {
                (void*)&alpha, (void*)&dA_array, (void*)&dA, &ldda, &strideA,
                               (void*)&dx_array, (void*)&dx, &incx, &stridex,
                (void*)&beta,         &dy_array,        &dy, &incy, &stridey, &batchCount};
        cudaError_t e = cudaLaunchKernel((void*)dgemvn_batched_smallsq_kernel<double, N>, grid, threads, kernel_args, shmem, queue->cuda_stream());
        if( e != cudaSuccess ) {
            return -100;
        }
    }
    else {
        void *kernel_args[] = {
                &transA,
                (void*)&alpha, (void*)&dA_array, (void*)&dA, &ldda, &strideA,
                               (void*)&dx_array, (void*)&dx, &incx, &stridex,
                (void*)&beta,         &dy_array,        &dy, &incy, &stridey, &batchCount};
        cudaError_t e = cudaLaunchKernel((void*)dgemvc_batched_smallsq_kernel<double, N>, grid, threads, kernel_args, shmem, queue->cuda_stream());
        if( e != cudaSuccess ) {
            return -100;
        }
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
static magma_int_t
dgemv_batched_smallsq_core(
    magma_trans_t transA, magma_int_t n,
    const double alpha,
    double const * const * dA_array, const double* dA, magma_int_t ldda, magma_int_t strideA,
    double const * const * dx_array, const double* dx, magma_int_t incx, magma_int_t stridex,
    const double beta,
    double**      dy_array,       double* dy, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    switch(n){
        case  1: info = dgemv_batched_smallsq_kernel_driver< 1>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  2: info = dgemv_batched_smallsq_kernel_driver< 2>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  3: info = dgemv_batched_smallsq_kernel_driver< 3>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  4: info = dgemv_batched_smallsq_kernel_driver< 4>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  5: info = dgemv_batched_smallsq_kernel_driver< 5>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  6: info = dgemv_batched_smallsq_kernel_driver< 6>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  7: info = dgemv_batched_smallsq_kernel_driver< 7>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  8: info = dgemv_batched_smallsq_kernel_driver< 8>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  9: info = dgemv_batched_smallsq_kernel_driver< 9>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 10: info = dgemv_batched_smallsq_kernel_driver<10>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 11: info = dgemv_batched_smallsq_kernel_driver<11>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 12: info = dgemv_batched_smallsq_kernel_driver<12>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 13: info = dgemv_batched_smallsq_kernel_driver<13>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 14: info = dgemv_batched_smallsq_kernel_driver<14>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 15: info = dgemv_batched_smallsq_kernel_driver<15>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 16: info = dgemv_batched_smallsq_kernel_driver<16>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 17: info = dgemv_batched_smallsq_kernel_driver<17>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 18: info = dgemv_batched_smallsq_kernel_driver<18>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 19: info = dgemv_batched_smallsq_kernel_driver<19>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 20: info = dgemv_batched_smallsq_kernel_driver<20>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 21: info = dgemv_batched_smallsq_kernel_driver<21>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 22: info = dgemv_batched_smallsq_kernel_driver<22>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 23: info = dgemv_batched_smallsq_kernel_driver<23>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 24: info = dgemv_batched_smallsq_kernel_driver<24>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 25: info = dgemv_batched_smallsq_kernel_driver<25>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 26: info = dgemv_batched_smallsq_kernel_driver<26>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 27: info = dgemv_batched_smallsq_kernel_driver<27>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 28: info = dgemv_batched_smallsq_kernel_driver<28>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 29: info = dgemv_batched_smallsq_kernel_driver<29>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 30: info = dgemv_batched_smallsq_kernel_driver<30>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 31: info = dgemv_batched_smallsq_kernel_driver<31>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 32: info = dgemv_batched_smallsq_kernel_driver<32>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        default:;
    }

    return info;
}

/******************************************************************************/
extern "C" magma_int_t
magmablas_dgemv_batched_smallsq(
    magma_trans_t transA, magma_int_t n,
    const double alpha,
    double const * const * dA_array, magma_int_t ldda,
    double const * const * dx_array, magma_int_t incx,
    const double beta,
    double**      dy_array, magma_int_t incy,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < n )
        info = -5;
    else if ( incx <= 0 )
        info = -7;
    else if ( incy <= 0 )
        info = -10;

    if (info != 0) {
        return info;
    }

    info = dgemv_batched_smallsq_core(
            transA, n,
            alpha, dA_array, NULL, ldda, 0,
                   dx_array, NULL, incx, 0,
            beta,  dy_array, NULL, incy, 0,
            batchCount, queue );

    return info;
}

/******************************************************************************/
extern "C" magma_int_t
magmablas_dgemv_batched_strided_smallsq(
    magma_trans_t transA, magma_int_t n,
    const double alpha,
    const double* dA, magma_int_t ldda, magma_int_t strideA,
    const double* dx, magma_int_t incx, magma_int_t stridex,
    const double beta,
    double* dy, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < n )
        info = -5;
    else if ( incx <= 0 )
        info = -7;
    else if ( incy <= 0 )
        info = -10;

    if (info != 0) {
        return info;
    }

    info = dgemv_batched_smallsq_core(
            transA, n,
            alpha, NULL, dA, ldda, strideA,
                   NULL, dx, incx, stridex,
            beta,  NULL, dy, incy, stridey,
            batchCount, queue );

    return info;
}
