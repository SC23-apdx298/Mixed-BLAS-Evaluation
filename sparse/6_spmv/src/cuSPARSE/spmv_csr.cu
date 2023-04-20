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
#include <cusparse.h>         // cusparseSpMV
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include <iostream>

#include "biio.h"
#include "power_nvml.hpp"

// #define COMPUTE_TYPE_AX CUDA_R_64F
// #define VALUE_TYPE_AX double
// #define COMPUTE_TYPE_Y CUDA_R_64F
// #define VALUE_TYPE_Y double
// #define COMPUTE_TYPE CUDA_R_64F
// #define ALPHA_TYPE double

#define WARMUP_TIMES 5
#define BENCH_REPEAT 20

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

__global__ void d_print_csr(int *row_ptr, int *col_idx, VALUE_TYPE_AX *val, int m)
{
    printf("device: \n");
    printf("row ptr: ");
    for (size_t i = 0; i <= m; i++)
    {
        printf("%d,", row_ptr[i]);
    }
    printf("\ncol idx: ");
    for (size_t i = 0; i < row_ptr[m]; i++)
    {
        printf("%d,", col_idx[i]);
    }
    printf("\n val:");
    for (size_t i = 0; i < row_ptr[m]; i++)
    {
        printf("%.1f,", val[i]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{
    // Host problem definition
    int A_num_rows;
    int A_num_cols;
    int A_nnz;

    ALPHA_TYPE alpha = 1.0f;
    ALPHA_TYPE beta = 0.0f;

    char *filename = argv[2];
    int device_id = atoi(argv[1]);
    cudaSetDevice(device_id);
    int isSymmetricA;

    int *hA_csrOffsets;
    int *hA_columns;
    double *hA_values_tmp;

    read_Dmatrix_32(&A_num_rows, &A_num_cols, &A_nnz, &hA_csrOffsets, &hA_columns, &hA_values_tmp, &isSymmetricA, filename);

    VALUE_TYPE_AX *hA_values = (VALUE_TYPE_AX *)malloc(A_nnz * sizeof(VALUE_TYPE_AX));

    VALUE_TYPE_AX *hX = (VALUE_TYPE_AX *)malloc(A_num_cols * sizeof(VALUE_TYPE_AX));

    for (size_t i = 0; i < A_num_cols; i++)
    {
        hX[i] = 1.0;
    }
    for (size_t i = 0; i < A_nnz; i++)
    {
        hA_values[i] = (VALUE_TYPE_AX)hA_values_tmp[i];
    }

    INFO_LOG("Init A\n");

    VALUE_TYPE_Y *hY = (VALUE_TYPE_Y *)malloc(A_num_rows * sizeof(VALUE_TYPE_Y));
    memset(hY, 0, A_num_rows * sizeof(VALUE_TYPE_Y));

    double *hY_result = (double *)malloc(A_num_rows * sizeof(double));
    memset(hY_result, 0, A_num_rows * sizeof(double));

    INFO_LOG("Init Y\n");

#ifdef DEBUG
    for (size_t i = 0; i < A_num_rows; i++)
    {
        for (size_t j = hA_csrOffsets[i]; j < hA_csrOffsets[i + 1]; j++)
        {
            hY_result[i] = hY_result[i] + (double)(hA_values[j]) * (double)(hX[hA_columns[j]]);
        }
    }
#endif

    //--------------------------------------------------------------------------
    // Device memory management
    int *dA_csrOffsets, *dA_columns;
    VALUE_TYPE_AX *dA_values, *dX, *dY;
    CHECK_CUDA(cudaMalloc((void **)&dA_csrOffsets,
                          (A_num_rows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dA_values, A_nnz * sizeof(VALUE_TYPE_AX)))
    CHECK_CUDA(cudaMalloc((void **)&dX, A_num_cols * sizeof(VALUE_TYPE_AX)))
    CHECK_CUDA(cudaMalloc((void **)&dY, A_num_rows * sizeof(VALUE_TYPE_Y)))

    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                          (A_num_rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(VALUE_TYPE_AX),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dX, hX, A_num_cols * sizeof(VALUE_TYPE_AX),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dY, hY, A_num_rows * sizeof(VALUE_TYPE_Y),
                          cudaMemcpyHostToDevice))
    INFO_LOG("Cuda mem init\n");

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void *dBuffer = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                     dA_csrOffsets, dA_columns, dA_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, COMPUTE_TYPE_AX))
    // d_print_csr<<<1,1>>>(dA_csrOffsets, dA_columns, dA_values, A_num_rows);
    cudaDeviceSynchronize();
    // Create dense vector X
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, A_num_cols, dX, COMPUTE_TYPE_AX))
    // Create dense vector y
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, A_num_rows, dY, COMPUTE_TYPE_Y))
    // allocate an external buffer if needed
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, COMPUTE_TYPE,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize))
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

    INFO_LOG("cuSPARSE init\n");

    // printf("buff size: %d\n", bufferSize);

    // warm up
    for (size_t i = 0; i < WARMUP_TIMES; i++)
    {
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, COMPUTE_TYPE,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
    }

    // execute SpMV
    struct timeval start, end;
    cudaDeviceSynchronize();
    // nvmlAPIRun();
    gettimeofday(&start, NULL);
    for (size_t i = 0; i < BENCH_REPEAT; i++)
    {
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, COMPUTE_TYPE,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer))
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    // nvmlAPIEnd();
    // double mwatt = get_avg_power_usage();
    double time_cusparse_spmv = ((end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0) / BENCH_REPEAT;
    // printf("%lu,%lu,%lu,%d,%d,%d,%.4lf\n", sizeof(VALUE_TYPE_AX), sizeof(VALUE_TYPE_Y), sizeof(VALUE_TYPE), A_num_rows, A_num_cols, A_nnz, time_cusparse_spmv);
    std::cout << sizeof(VALUE_TYPE_AX) << "," << sizeof(VALUE_TYPE_Y) << "," << sizeof(double) << "," << A_num_rows << "," << A_num_cols << "," << A_nnz << "," << time_cusparse_spmv << "," << (2 * (double)A_nnz) / (1.0e+6 * (double)time_cusparse_spmv);
    // destroy matrix/vector descriptors
    // std::cout  << "," << mwatt << std::endl;
    std::cout << std::endl;

    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX))
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY))
    CHECK_CUSPARSE(cusparseDestroy(handle))
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA(cudaMemcpy(hY, dY, A_num_rows * sizeof(VALUE_TYPE_Y),
                          cudaMemcpyDeviceToHost))

    int correct = 1;
#ifdef DEBUG
    for (int i = 0; i < A_num_rows; i++)
    {
        if ((double)(hY[i]) - hY_result[i] > 1e-2)
        { // direct floating point comparison is not
            printf("%d, %lf, %lf\n", i, (double)hY[i], hY_result[i]);
            correct = 0; // reliable
            break;
        }
    }
    if (correct)
    {
        // printf("spmv_csr_example test PASSED\n");
    }
    else
        printf("spmv_csr_example test FAILED: wrong result\n");
#endif
    // char out_filename[100];
    // for (size_t i = strlen(argv[1]); i > 0; i--)
    // {
    //     if (argv[1][i - 1] == '/')
    //     {
    //         sprintf(out_filename, "data/%s_%u.txt", argv[1] + i, sizeof(ALPHA_TYPE));
    //         break;
    //     }
    // }
    // vec_write_file(hY, A_num_rows, out_filename);

    //--------------------------------------------------------------------------

    // device memory deallocation
    CHECK_CUDA(cudaFree(dBuffer))
    CHECK_CUDA(cudaFree(dA_csrOffsets))
    CHECK_CUDA(cudaFree(dA_columns))
    CHECK_CUDA(cudaFree(dA_values))
    CHECK_CUDA(cudaFree(dX))
    CHECK_CUDA(cudaFree(dY))

    free(hA_csrOffsets);
    free(hA_columns);
    free(hA_values_tmp);
    free(hA_values);
    free(hX);
    free(hY);
    free(hY_result);

    return EXIT_SUCCESS;
}
