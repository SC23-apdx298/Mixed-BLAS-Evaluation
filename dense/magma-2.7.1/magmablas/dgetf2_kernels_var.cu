/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @author Ahmad Abdelfattah

       @generated from magmablas/zgetf2_kernels_var.cu, normal z -> d, Wed Feb 22 15:21:33 2023
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"
#include "shuffle.cuh"

#define PRECISION_d
#include "dgetf2_devicefunc.cuh"

/******************************************************************************/
__global__ void
idamax_kernel_vbatched(
        int length, magma_int_t *M, magma_int_t *N,
        double **dA_array, int Ai, int Aj, magma_int_t *ldda,
        magma_int_t** ipiv_array, int ipiv_i,
        magma_int_t *info_array, int step, int gbstep )
{
    extern __shared__ double sdata[];

    const int batchid = blockIdx.x;

    // compute the actual length
    int my_M    = (int)M[batchid];
    int my_N    = (int)N[batchid];
    int my_ldda = (int)ldda[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_M <= Ai || my_N <= Aj ) return;

    // compute the length of the vector for each matrix
    my_M -= Ai;
    my_M  = min(my_M, length);

    double *dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t *ipiv = ipiv_array[batchid] + ipiv_i;
    int tx = threadIdx.x;

    double *shared_x = sdata;
    int *shared_idx = (int*)(shared_x + zamax);

    idamax_devfunc(my_M, dA, 1, shared_x, shared_idx);

    if (tx == 0) {
        *ipiv = shared_idx[0] + step + 1; // Fortran Indexing & adjust pivot
        if (shared_x[0] == MAGMA_D_ZERO) {
            info_array[batchid] = shared_idx[0] + step + gbstep + 1;
        }
    }
}

/******************************************************************************/
extern "C" magma_int_t
magma_idamax_vbatched(
        magma_int_t length, magma_int_t *M, magma_int_t *N,
        double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t** ipiv_array, magma_int_t ipiv_i,
        magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
        magma_int_t batchCount, magma_queue_t queue)
{
    dim3 grid(batchCount, 1, 1);
    dim3 threads(zamax, 1, 1);

    idamax_kernel_vbatched<<< grid, threads, zamax * (sizeof(double) + sizeof(int)), queue->cuda_stream() >>>
    (length, M, N, dA_array, Ai, Aj, ldda, ipiv_array, ipiv_i, info_array, step, gbstep );

    return 0;
}

/******************************************************************************/
__global__
void dswap_kernel_vbatched(
        int max_n, magma_int_t *M, magma_int_t *N,
        double **dA_array, int Ai, int Aj, magma_int_t* ldda,
        magma_int_t** ipiv_array, int piv_adjustment)
{
    const int batchid = blockIdx.x;
    const int my_ldda = (int)ldda[batchid];
    int my_M          = (int)M[batchid];
    int my_N          = (int)N[batchid];
    int my_minmn      = min(my_M, my_N);

    // check if offsets produce out-of-bound pointers
    if( my_M <= Ai || my_N <= Aj || my_minmn <= Ai ) return;

    my_N -= Aj; // this is the maximum possible width
    my_N = min(my_N, max_n);

    // read the pivot entry at Ai
    magma_int_t *ipiv = ipiv_array[batchid] + Ai;
    __shared__ int jp;
    if (threadIdx.x == 0){
        jp  = ipiv[0] - 1; // roll-back Fortran indexing
        // magma_idamax_vbatched adjusts the pivot, so roll it back
        // because Ai and Aj are offsets that already take care of that
        jp -= piv_adjustment;
    }
    __syncthreads();

    if (jp == 0) return; // no swapping required

    double *dA  = dA_array[batchid] + Aj * my_ldda + Ai;
    double *dA1 = dA;
    double *dA2 = dA + jp;

    dswap_device_v2(my_N, dA1, my_ldda, dA2, my_ldda );
}

/******************************************************************************/
extern "C" magma_int_t
magma_dswap_vbatched(
        magma_int_t max_n, magma_int_t *M, magma_int_t *N,
        double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t** ipiv_array, magma_int_t piv_adjustment,
        magma_int_t batchCount, magma_queue_t queue)
{
    dim3 grid(batchCount, 1, 1);
    dim3 threads(zamax, 1, 1);

    dswap_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>
    (max_n, M, N, dA_array, Ai, Aj, ldda, ipiv_array, piv_adjustment);

    return 0;
}

/******************************************************************************/
__global__
void dscal_dger_1d_generic_kernel_vbatched(
        int max_m, int max_n,
        magma_int_t *M, magma_int_t *N,
        double **dA_array, int Ai, int Aj, magma_int_t *ldda,
        magma_int_t *info_array, int step, int gbstep)
{
    const int batchid = blockIdx.z;
    int my_M    = (int)M[batchid];
    int my_N    = (int)N[batchid];
    int my_ldda = (int)ldda[batchid];

    if( my_M <= Ai || my_N <= Aj ) return;
    my_M -= Ai; // this is the largest possible m per matrix
    my_N -= Aj; // this is the largest possible n per matrix

    my_M = min(my_M, max_m);
    my_N = min(my_N, max_n);

    double* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t *info = &info_array[batchid];
    dscal_dger_generic_device(my_M, my_N, dA, my_ldda, info, step, gbstep);
}


/******************************************************************************/
extern "C"
magma_int_t magma_dscal_dger_vbatched(
        magma_int_t max_M, magma_int_t max_N,
        magma_int_t *M, magma_int_t *N,
        double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
        magma_int_t batchCount, magma_queue_t queue)
{
    /*
    Specialized kernel which merged dscal and dger the two kernels
    1) dscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a dger Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);
    */

    magma_int_t max_batchCount = queue->get_maxBatch();
    const int tbx = 256;
    dim3 threads(tbx, 1, 1);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid(magma_ceildiv(max_M,tbx), 1, ibatch);

        dscal_dger_1d_generic_kernel_vbatched<<<grid, threads, 0, queue->cuda_stream()>>>
        (max_M, max_N, M+i, N+i, dA_array+i, Ai, Aj, ldda+i, info_array+i, step, gbstep);
    }
    return 0;
}

/******************************************************************************/
#define dA(i,j)              dA[(j) * my_ldda + (i)]
#define sA(i,j)              sA[(j) * my_M + (i)]
__global__
void
dgetf2_fused_sm_kernel_vbatched(
        int max_M, int max_N, int max_minMN, int max_MxN,
        magma_int_t *M, magma_int_t *N,
        double** dA_array, int Ai, int Aj, magma_int_t* ldda,
        magma_int_t** dipiv_array, int ipiv_i,
        magma_int_t *info,  int gbstep, int batchCount )
{
    extern __shared__ double zdata[];
    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int ntx     = blockDim.x;
    const int batchid = (blockIdx.x * blockDim.y) + ty;
    if(batchid >= batchCount) return;

    // read data of assigned problem
    int my_M         = (int)M[batchid];
    int my_N         = (int)N[batchid];
    int my_ldda      = (int)ldda[batchid];
    int my_minmn     = min(my_M, my_N);
    double* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t* dipiv     = dipiv_array[batchid] + ipiv_i;

    // check offsets
    if( my_M <= Ai || my_N <= Aj || my_minmn <= ipiv_i ) return;
    my_M     -= Ai;
    my_N     -= Aj;
    my_M      = min(my_M, max_M);
    my_N      = min(my_N, max_N);
    my_minmn  = min(my_M, my_N);

    double *sA = (double*)(zdata);
    double* dsx = (double*)(sA + blockDim.y * max_MxN);
    int* isx    = (int*)(dsx + blockDim.y * max_M);
    int* sipiv  = (int*)(isx + blockDim.y * max_M);
    dsx   += ty * max_M;
    isx   += ty * max_M;
    sipiv += ty * max_minMN;

    double reg  = MAGMA_D_ZERO;
    double rTmp = MAGMA_D_ZERO;

    int max_id;
    int linfo = (gbstep == 0) ? 0 : *info;
    double rx_abs_max = MAGMA_D_ZERO;

    // init sipiv
    for(int i = tx; i < my_minmn; i+=ntx) {
        sipiv[i] = 0;
    }

    // read
    for(int j = 0; j < my_N; j++){
        for(int i = tx; i < my_M; i+=ntx) {
            sA(i,j) = dA(i,j);
        }
    }
    __syncthreads();

    for(int j = 0; j < my_minmn; j++){
        // idamax and find pivot
        for(int i = j+tx; i < my_M; i+=ntx) {
            dsx[ i ] = fabs(MAGMA_D_REAL( sA(i,j) )) + fabs(MAGMA_D_IMAG( sA(i,j) ));
            isx[ i ] = i-j;
        }
        __syncthreads();
        magma_getidmax_n(my_M-j, tx, dsx+j, isx+j);
        // the above devfunc has syncthreads at the end
        rx_abs_max = dsx[j];
        max_id     = j + isx[j];
        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (gbstep+j+1) : linfo;
        if( tx == 0 ) sipiv[ j ] = max_id;
        __syncthreads();


        // swap
        if(max_id != j) {
            for(int i = tx; i < my_N; i+=ntx) {
                reg          = sA(j     ,i);
                sA(j,i)      = sA(max_id,i);
                sA(max_id,i) = reg;
            }
        }
        __syncthreads();

        if( linfo == 0 ) {
            reg = MAGMA_D_DIV( MAGMA_D_ONE, sA(j,j) );
            for(int i = (tx+j+1); i < my_M; i+=ntx) {
                rTmp    = reg * sA(i,j);
                sA(i,j) = rTmp;
                for(int jj = j+1; jj < my_N; jj++) {
                    sA(i,jj) -= rTmp * sA(j,jj);
                }
            }
        }
        __syncthreads();

    }

    if(tx == 0){
        (*info) = (magma_int_t)( linfo );
    }

    // write pivot
    for(int i = tx; i < my_minmn; i+=ntx) {
        dipiv[i] = (magma_int_t)(sipiv[i] + 1);
    }

    // write A
    for(int j = 0; j < my_N; j++) {
        for(int i = tx; i < my_M; i+=ntx) {
            dA(i,j) = sA(i,j);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_dgetf2_fused_sm_vbatched(
    magma_int_t max_M, magma_int_t max_N, magma_int_t max_minMN, magma_int_t max_MxN,
    magma_int_t* m, magma_int_t* n,
    double** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t** dipiv_array, magma_int_t ipiv_i,
    magma_int_t* info_array, magma_int_t gbstep,
    magma_int_t nthreads, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_device_t device;
    magma_getdevice( &device );

    nthreads = nthreads <= 0 ? (max_M/2) : nthreads;
    #ifdef MAGMA_HAVE_CUDA
    nthreads = magma_roundup(nthreads, 32);
    #else
    nthreads = magma_roundup(nthreads, 64);
    #endif
    nthreads = min(nthreads, 1024);

    // in a variable-size setting, setting ntcol > 1 may lead to
    // kernel deadlocks due to different thread-groups calling
    // syncthreads at different points
    const magma_int_t ntcol = 1;
    int         shmem = ( max_MxN   * sizeof(double) );
    shmem            += ( max_M     * sizeof(double) );
    shmem            += ( max_M     * sizeof(int) );
    shmem            += ( max_minMN * sizeof(int) );
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
        cudaFuncSetAttribute(dgetf2_fused_sm_kernel_vbatched, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
        arginfo = -100;
        return arginfo;
    }

    if( check_launch_only == 1 ) return arginfo;

    void *kernel_args[] = {&max_M, &max_N, &max_minMN, &max_MxN, &m, &n, &dA_array, &Ai, &Aj, &ldda, &dipiv_array, &ipiv_i, &info_array, &gbstep, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)dgetf2_fused_sm_kernel_vbatched, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}


/******************************************************************************/
#define SLDA(n)              ( (((n)+1)%4) == 0 ? (n) : (n+1) )
#define ibatch    (0)
template<int max_N>
__global__ void
dgetf2_fused_kernel_vbatched(
        int max_M,
        magma_int_t* M, magma_int_t* N,
        double** dA_array, int Ai, int Aj, magma_int_t* ldda,
        magma_int_t** dipiv_array, int ipiv_i,
        magma_int_t* info_array, int batchCount)
{
    extern __shared__ double data[];
    const int tx = threadIdx.x;
    const int batchid = blockIdx.x * blockDim.y + threadIdx.y;
    if(batchid >= batchCount)return;

    // read data of assigned problem
    int my_M         = (int)M[batchid];
    int my_N         = (int)N[batchid];
    int my_ldda      = (int)ldda[batchid];
    int my_minmn     = (int)min(my_M, my_N);
    double* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t* dipiv     = dipiv_array[batchid] + ipiv_i;

    // check offsets
    if( my_M <= Ai || my_N <= Aj || my_minmn <= ipiv_i ) return;
    // (my_M, my_N) based on (M,N) and offsets (Ai,Aj)
    my_M     -= Ai;
    my_N     -= Aj;

    // now compare (my_M,my_N) with max_M, max_N
    my_M = min(my_M, max_M);
    my_N = min(my_N, max_N);
    my_minmn  = min(my_M, my_N);

    int rowid, gbstep = Aj;
    int orginfo = (gbstep == 0) ? 0 : info_array[batchid];
    int linfo   = 0;
    const int slda = SLDA(max_M);
    double  rA[max_N] = {MAGMA_D_ZERO};

    // init sA into identity
    double* sA = (double*)data;
    #pragma unroll
    for(int j = 0; j < max_N; j++) {
        sA[j * slda + tx] = MAGMA_D_ZERO;
    }

    // read A into sm then mv to reg
    if(tx < my_M) {
        for(int j = 0; j < my_N; j++) {
            sA[j * slda + tx] = dA[j * my_ldda + tx];
        }
    }
    __syncthreads();

    #pragma unroll
    for(int j = 0; j < max_N; j++){
        rA[j] = sA[ j * slda + tx ];
    }
    __syncthreads();

    dgetf2_fused_device<max_N>(
             max_M, my_minmn, rA,
             dipiv,
             sA, linfo, gbstep, rowid);

    __syncthreads();

    // write to shared
    #pragma unroll
    for(int j = 0; j < max_N; j++){
        sA[ j * slda + rowid ] = rA[j];
    }
    __syncthreads();

    // ignore any info beyond minmn
    // (meaning singularity is encountered at the padded matrix)
    linfo = (linfo >= my_minmn) ? 0 : linfo;
    linfo = (orginfo == 0) ? linfo : orginfo;

    if(tx == 0){
        info_array[batchid] = (magma_int_t)( linfo );
    }

    // write to global
    if(tx < my_M) {
        for(int j = 0; j < my_N; j++) {
            dA[j * my_ldda + tx] = sA[j * slda + tx];
        }
    }
}

/******************************************************************************/
template<int max_N>
static magma_int_t
magma_dgetf2_fused_kernel_driver_vbatched(
    magma_int_t max_M,
    magma_int_t* M, magma_int_t* N,
    double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t **dipiv_array, magma_int_t ipiv_i,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_device_t device;
    magma_getdevice( &device );

    // this kernel works only if m <= n for every matrix
    // this is only for short-wide sizes that fit in shared memory
    // should not affect performance for other shapes
    max_M = max(max_M, max_N);

    int ntcol = 1;
    int shmem = 0, shmem_1 = 0, shmem_2 = 0;
    shmem_1 += max_N * sizeof(double);
    shmem_1 += max_M * sizeof(double);
    shmem_1 += max_M * sizeof(int);    // not magma_int_t
    shmem_1 += max_N * sizeof(int);    // not magma_int_t

    shmem_2 += SLDA(max_M) * max_N * sizeof(double);

    //printf("max_M = %d, max_N = %d\n", max_M, max_N);
    //printf("shmem-1 = %d, shmem-2 = %d\n", shmem_1, shmem_2);

    shmem  = max(shmem_1, shmem_2);
    shmem *= ntcol;

    dim3 grid(magma_ceildiv(batchCount,ntcol), 1, 1);
    dim3 threads(max_M, ntcol, 1);

    // get max. dynamic shared memory on the GPU
    int nthreads_max, nthreads = max_M * ntcol, shmem_max = 0;
    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(dgetf2_fused_kernel_vbatched<max_N>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
        arginfo = -100;
        return arginfo;
    }

    void *kernel_args[] = {&max_M, &M, &N, &dA_array, &Ai, &Aj, &ldda, &dipiv_array, &ipiv_i, &info_array, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)dgetf2_fused_kernel_vbatched<max_N>, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}

/******************************************************************************/
extern "C" magma_int_t
magma_dgetf2_fused_vbatched(
    magma_int_t max_M, magma_int_t max_N,
    magma_int_t max_minMN, magma_int_t max_MxN,
    magma_int_t* M, magma_int_t* N,
    double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t **dipiv_array, magma_int_t ipiv_i,
    magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue)
{
    //printf("max_M = %d, max_N = %d\n", max_M, max_N);

    magma_int_t info = 0;
    if(max_M < 0 ) {
        info = -1;
    }
    else if(max_N < 0){
        info = -2;
    }

    if(info < 0) return info;


    info = -1; // init a negative value
    switch(max_N) {
        #if 1
        case  1: info = magma_dgetf2_fused_kernel_driver_vbatched< 1>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  2: info = magma_dgetf2_fused_kernel_driver_vbatched< 2>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  3: info = magma_dgetf2_fused_kernel_driver_vbatched< 3>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  4: info = magma_dgetf2_fused_kernel_driver_vbatched< 4>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  5: info = magma_dgetf2_fused_kernel_driver_vbatched< 5>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  6: info = magma_dgetf2_fused_kernel_driver_vbatched< 6>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  7: info = magma_dgetf2_fused_kernel_driver_vbatched< 7>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  8: info = magma_dgetf2_fused_kernel_driver_vbatched< 8>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  9: info = magma_dgetf2_fused_kernel_driver_vbatched< 9>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 10: info = magma_dgetf2_fused_kernel_driver_vbatched<10>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 11: info = magma_dgetf2_fused_kernel_driver_vbatched<11>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 12: info = magma_dgetf2_fused_kernel_driver_vbatched<12>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 13: info = magma_dgetf2_fused_kernel_driver_vbatched<13>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 14: info = magma_dgetf2_fused_kernel_driver_vbatched<14>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 15: info = magma_dgetf2_fused_kernel_driver_vbatched<15>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 16: info = magma_dgetf2_fused_kernel_driver_vbatched<16>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 17: info = magma_dgetf2_fused_kernel_driver_vbatched<17>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 18: info = magma_dgetf2_fused_kernel_driver_vbatched<18>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 19: info = magma_dgetf2_fused_kernel_driver_vbatched<19>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 20: info = magma_dgetf2_fused_kernel_driver_vbatched<20>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 21: info = magma_dgetf2_fused_kernel_driver_vbatched<21>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 22: info = magma_dgetf2_fused_kernel_driver_vbatched<22>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 23: info = magma_dgetf2_fused_kernel_driver_vbatched<23>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 24: info = magma_dgetf2_fused_kernel_driver_vbatched<24>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 25: info = magma_dgetf2_fused_kernel_driver_vbatched<25>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 26: info = magma_dgetf2_fused_kernel_driver_vbatched<26>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 27: info = magma_dgetf2_fused_kernel_driver_vbatched<27>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 28: info = magma_dgetf2_fused_kernel_driver_vbatched<28>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 29: info = magma_dgetf2_fused_kernel_driver_vbatched<29>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 30: info = magma_dgetf2_fused_kernel_driver_vbatched<30>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 31: info = magma_dgetf2_fused_kernel_driver_vbatched<31>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 32: info = magma_dgetf2_fused_kernel_driver_vbatched<32>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        #endif
        default: ;
    }

    //printf("info from reg = %d\n", info);

    if( info != 0 ) {
        // try sm version
        magma_int_t sm_nthreads = max(32, max_M / 2);
        sm_nthreads = magma_roundup(sm_nthreads, 32);
        info = magma_dgetf2_fused_sm_vbatched(
                    max_M, max_N, max_minMN, max_MxN,
                    M, N, dA_array, Ai, Aj, ldda,
                    dipiv_array, ipiv_i,
                    info_array, Aj, sm_nthreads, 0, batchCount, queue );
    }

    return info;
}
