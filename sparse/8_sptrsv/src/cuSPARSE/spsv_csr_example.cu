/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
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
#include <cusparse.h>         // cusparseSpSV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include <sys/time.h>

#include "common.h"
#include "biio.h"

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
    // Host problem definition
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    int A_num_rows;
    int A_num_cols;
    int A_nnz;

    VALUE_TYPE alpha = 1.0f;

    char *filename = argv[1];
    int isSymmetricA;

    int *hA_csrOffsets;
    int *hA_columns;
    double *hA_values_tmp;
    read_Dmatrix_32(&A_num_rows, &A_num_cols, &A_nnz, &hA_csrOffsets, &hA_columns, &hA_values_tmp, &isSymmetricA, filename);

    // printf("%d,%d,%d,", sizeof(VALUE_TYPE), sizeof(VALUE_TYPE), sizeof(VALUE_TYPE));
    // printf("%d,%d,%d,", A_num_rows, A_num_cols, A_nnz);

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
    INFO_LOG("A complete\n");
    free(hA_values_tmp);

    VALUE_TYPE *hX = (VALUE_TYPE *)malloc(A_num_cols * sizeof(VALUE_TYPE));

    for (size_t i = 0; i < A_num_rows; i++)
    {
        hX[i] = 1.0;
    }
    INFO_LOG("X complete\n");

    VALUE_TYPE *hY = (VALUE_TYPE *)malloc(A_num_rows * sizeof(VALUE_TYPE));
    memset(hY, 0, A_num_rows * sizeof(VALUE_TYPE));

    VALUE_TYPE *hY_result = (VALUE_TYPE *)malloc(A_num_rows * sizeof(VALUE_TYPE));
    memset(hY_result, 0, A_num_rows * sizeof(VALUE_TYPE));

    int nnzTR;
    int *csrRowPtrTR;
    int *csrColIdxTR;
    VALUE_TYPE *csrValTR;
    // extract L and U with a unit diagonal of A
    csrRowPtrTR = (int *)malloc((A_num_rows + 1) * sizeof(int));
    csrColIdxTR = (int *)malloc((A_num_rows + A_nnz) * sizeof(int));
    csrValTR = (VALUE_TYPE *)malloc((A_num_rows + A_nnz) * sizeof(VALUE_TYPE));

    int nnz_pointer = 0;
    csrRowPtrTR[0] = 0;
    for (int i = 0; i < A_num_rows; i++)
    {
        for (int j = hA_csrOffsets[i]; j < hA_csrOffsets[i + 1]; j++)
        {
            if (hA_columns[j] < i)
            {
                csrColIdxTR[nnz_pointer] = hA_columns[j];
                csrValTR[nnz_pointer] = hA_values[j];
                nnz_pointer++;
            }
        }
        // add dia nonzero
        csrColIdxTR[nnz_pointer] = i;
        csrValTR[nnz_pointer] = 1.0;
        nnz_pointer++;
        csrRowPtrTR[i + 1] = nnz_pointer;
    }
    INFO_LOG("TR complete\n");

    int nnz_tmp = csrRowPtrTR[A_num_rows];
    nnzTR = nnz_tmp;

    csrColIdxTR = (int *)realloc(csrColIdxTR, sizeof(int) * nnzTR);
    csrValTR = (VALUE_TYPE *)realloc(csrValTR, sizeof(VALUE_TYPE) * nnzTR);
    INFO_LOG("realloc complete\n");

    //--------------------------------------------------------------------------
    // Device memory management
    int *dA_csrOffsets, *dA_columns;
    VALUE_TYPE *dA_values, *dX, *dY;
    CHECK_CUDA(cudaMalloc((void **)&dA_csrOffsets,
                          (A_num_rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dA_columns, nnzTR * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dA_values, nnzTR * sizeof(VALUE_TYPE)))
    CHECK_CUDA(cudaMalloc((void **)&dX, A_num_cols * sizeof(VALUE_TYPE)))
    CHECK_CUDA(cudaMalloc((void **)&dY, A_num_rows * sizeof(VALUE_TYPE)))

    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, csrRowPtrTR,
                          (A_num_rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_columns, csrColIdxTR, nnzTR * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_values, csrValTR, nnzTR * sizeof(VALUE_TYPE),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dX, hX, A_num_cols * sizeof(VALUE_TYPE),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dY, hY, A_num_rows * sizeof(VALUE_TYPE),
                          cudaMemcpyHostToDevice))
    INFO_LOG("device mem complete\n");
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    cusparseSpSVDescr_t spsvDescr;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, nnzTR,
                                     dA_csrOffsets, dA_columns, dA_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, COMPUTE_TYPE))
    // Create dense vector X
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, COMPUTE_TYPE))
    // Create dense vector y
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, COMPUTE_TYPE))
    // Create opaque data structure, that holds analysis data between calls.
    CHECK_CUSPARSE(cusparseSpSV_createDescr(&spsvDescr))
    // Specify Lower|Upper fill mode.
    cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE,
                                             &fillmode, sizeof(fillmode)))
    // Specify Unit|Non-Unit diagonal type.
    cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CHECK_CUSPARSE(cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE,
                                             &diagtype, sizeof(diagtype)))
    // allocate an external buffer for analysis
    CHECK_CUSPARSE(cusparseSpSV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, vecY, COMPUTE_TYPE,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr,
        &bufferSize))
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))
    CHECK_CUSPARSE(cusparseSpSV_analysis(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, vecY, COMPUTE_TYPE,
        CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, dBuffer))

    INFO_LOG("warming up 10 times\n");

    // execute
    for (size_t i = 0; i < 10; i++)
    {
        CHECK_CUSPARSE(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha, matA, vecX, vecY, COMPUTE_TYPE,
                                          CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr))
    }
    // benckmark SpSV
    // cudaEventRecord(start, NULL);

    struct timeval start, end;
    cudaDeviceSynchronize();
    INFO_LOG("benching %d times\n", BENCH_REPEAT);
    gettimeofday(&start, NULL);
    for (size_t i = 0; i < BENCH_REPEAT; i++)
    {
        CHECK_CUSPARSE(cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha, matA, vecX, vecY, COMPUTE_TYPE,
                                          CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr))
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    // cudaEventRecord(stop, NULL);
    // cudaEventSynchronize(stop);

    double time = ((end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0) / BENCH_REPEAT;
    // cudaEventElapsedTime(&time, start, stop);
    printf("%d,%d,%d,%d,%d,%d,%f,%f\n", sizeof(VALUE_TYPE), sizeof(VALUE_TYPE), sizeof(VALUE_TYPE), A_num_rows, A_num_cols, nnzTR, time, (2 * nnzTR) / (time * 1e6));

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
    CHECK_CUSPARSE(cusparseSpSV_destroyDescr(spsvDescr));
    CHECK_CUSPARSE(cusparseDestroy(handle))
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA(cudaMemcpy(hY, dY, A_num_rows * sizeof(float),
                          cudaMemcpyDeviceToHost))
    int correct = 1;
    // for (int i = 0; i < A_num_rows; i++)
    // {
    //     if (hY[i] != hY_result[i])
    //     {                // direct floating point comparison is not
    //         correct = 0; // reliable
    //         break;
    //     }
    // }
    // if (correct)
    //     printf("spsv_csr_example test PASSED\n");
    // else
    //     printf("spsv_csr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(dA_csrOffsets))
    CHECK_CUDA(cudaFree(dA_columns))
    CHECK_CUDA(cudaFree(dA_values))
    CHECK_CUDA(cudaFree(dX))
    CHECK_CUDA(cudaFree(dY))
    

    // host memory deallocation
    free(hA_csrOffsets);
    free(hA_columns);
    free(hA_values);
    free(hX);
    free(hY);
    free(hY_result);

    return EXIT_SUCCESS;
}
