/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpGEMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <sys/time.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "biio.h"

#ifndef BENCH_REPEAT
#define BENCH_REPEAT 100
#endif

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    }

int main(int argc, char **argv)
{
    int A_num_rows;
    int A_num_cols;
    int A_nnz;
    int C_nnz;

    VALUE_TYPE alpha = 1.0f;
    VALUE_TYPE beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = COMPUTE_TYPE;
    char *filename = argv[2];
    int device_id = atoi(argv[1]);
    cudaSetDevice(device_id);
    int isSymmetricA;

    int *hA_csrOffsets;
    int *hA_columns;
    double *hA_values_tmp;
    read_Dmatrix_32(&A_num_rows, &A_num_cols, &A_nnz, &hA_csrOffsets, &hA_columns, &hA_values_tmp, &isSymmetricA, filename);

    if (A_num_rows != A_num_cols)
    {
        free(hA_csrOffsets);
        free(hA_columns);
        free(hA_values_tmp);
        printf("mat must be square, exit!\n");
        return -1;
    }

    VALUE_TYPE *hA_values = (VALUE_TYPE *)malloc(A_nnz * sizeof(VALUE_TYPE));

    for (size_t i = 0; i < A_nnz; i++)
    {
        hA_values[i] = (VALUE_TYPE)(hA_values_tmp[i]);
    }
    free(hA_values_tmp);

    int B_num_rows = A_num_rows;
    int B_num_cols = A_num_cols;
    int B_nnz = A_nnz;
    int *hB_csrOffsets = hA_csrOffsets;
    int *hB_columns = hA_columns;
    VALUE_TYPE *hB_values = hA_values;

    // int *hC_csrOffsets = (int *)malloc((A_num_rows + 1) * sizeof(int));
    // int *hC_columns;
    // VALUE_TYPE hC_values;

    //--------------------------------------------------------------------------
    // Device memory management: Allocate and copy A, B
    int *dA_csrOffsets, *dA_columns, *dB_csrOffsets, *dB_columns,
        *dC_csrOffsets, *dC_columns;
    VALUE_TYPE *dA_values, *dB_values, *dC_values;
    // allocate A
    CHECK_CUDA(cudaMalloc((void **)&dA_csrOffsets,
                          (A_num_rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dA_values, A_nnz * sizeof(VALUE_TYPE)))
    // allocate B
    CHECK_CUDA(cudaMalloc((void **)&dB_csrOffsets,
                          (B_num_rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dB_columns, B_nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dB_values, B_nnz * sizeof(VALUE_TYPE)))
    // allocate C offsets
    CHECK_CUDA(cudaMalloc((void **)&dC_csrOffsets,
                          (A_num_rows + 1) * sizeof(int)))

    // copy A
    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                          (A_num_rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values,
                          A_nnz * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice))
    // copy B
    CHECK_CUDA(cudaMemcpy(dB_csrOffsets, hB_csrOffsets,
                          (B_num_rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB_columns, hB_columns, B_nnz * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dB_values, hB_values,
                          B_nnz * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice))
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                     dA_csrOffsets, dA_columns, dA_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, COMPUTE_TYPE))
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_num_rows, B_num_cols, B_nnz,
                                     dB_csrOffsets, dB_columns, dB_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, COMPUTE_TYPE))
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, A_num_rows, B_num_cols, 0,
                                     NULL, NULL, NULL,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, COMPUTE_TYPE))
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL))
    CHECK_CUDA(cudaMalloc((void **)&dBuffer1, bufferSize1))
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, dBuffer1))

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL))
    CHECK_CUDA(cudaMalloc((void **)&dBuffer2, bufferSize2))

    // warm up
    for (size_t i = 0; i < 10; i++)
    {
        CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                                              &alpha, matA, matB, &beta, matC,
                                              computeType, CUSPARSE_SPGEMM_DEFAULT,
                                              spgemmDesc, &bufferSize2, dBuffer2))
    }

    struct timeval start, end;
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);
    // bench
    for (size_t i = 0; i < BENCH_REPEAT; i++)
    {
        CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                                              &alpha, matA, matB, &beta, matC,
                                              computeType, CUSPARSE_SPGEMM_DEFAULT,
                                              spgemmDesc, &bufferSize2, dBuffer2))
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    double time_cusparse_spgemm = ((end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0) / BENCH_REPEAT;
    printf("%lu,%lu,%lu,%d,%d,%d,%lf\n", sizeof(VALUE_TYPE), sizeof(VALUE_TYPE), sizeof(VALUE_TYPE), A_num_rows, A_num_cols, A_nnz, time_cusparse_spgemm);

    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                        &C_nnz1))
    // allocate matrix C
    CHECK_CUDA(cudaMalloc((void **)&dC_columns, C_nnz1 * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dC_values, C_nnz1 * sizeof(VALUE_TYPE)))

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values))

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc))

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc))
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroySpMat(matB))
    CHECK_CUSPARSE(cusparseDestroySpMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))
    //--------------------------------------------------------------------------
    // device result check
    int *hC_csrOffsets_tmp = (int *)malloc((A_num_rows + 1) * sizeof(int));
    int *hC_columns_tmp = (int *)malloc((C_nnz1 + 1) * sizeof(int));
    VALUE_TYPE *hC_values_tmp = (VALUE_TYPE *)malloc((C_nnz1 + 1) * sizeof(VALUE_TYPE));
    CHECK_CUDA(cudaMemcpy(hC_csrOffsets_tmp, dC_csrOffsets,
                          (A_num_rows + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(hC_columns_tmp, dC_columns, C_nnz1 * sizeof(int),
                          cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(hC_values_tmp, dC_values, C_nnz1 * sizeof(VALUE_TYPE),
                          cudaMemcpyDeviceToHost))
    int correct = 1;
    // for (int i = 0; i < A_num_rows + 1; i++)
    // {
    //     if (hC_csrOffsets_tmp[i] != hC_csrOffsets[i])
    //     {
    //         correct = 0;
    //         break;
    //     }
    // }
    // for (int i = 0; i < C_nnz; i++)
    // {
    //     if (hC_columns_tmp[i] != hC_columns[i] ||
    //         hC_values_tmp[i] != hC_values[i])
    //     {                // direct VALUE_TYPEing point
    //         correct = 0; // comparison is not reliable
    //         break;
    //     }
    // }
    // if (correct)
    //     printf("spgemm_example test PASSED\n");
    // else
    // {
    //     printf("spgemm_example test FAILED: wrong result\n");
    //     return EXIT_FAILURE;
    // }
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer1))
    CHECK_CUDA(cudaFree(dBuffer2))
    CHECK_CUDA(cudaFree(dA_csrOffsets))
    CHECK_CUDA(cudaFree(dA_columns))
    CHECK_CUDA(cudaFree(dA_values))
    CHECK_CUDA(cudaFree(dB_csrOffsets))
    CHECK_CUDA(cudaFree(dB_columns))
    CHECK_CUDA(cudaFree(dB_values))
    CHECK_CUDA(cudaFree(dC_csrOffsets))
    CHECK_CUDA(cudaFree(dC_columns))
    CHECK_CUDA(cudaFree(dC_values))

    // host memory deallocation
    free(hA_csrOffsets);
    free(hA_columns);
    free(hA_values);

    return EXIT_SUCCESS;
}
