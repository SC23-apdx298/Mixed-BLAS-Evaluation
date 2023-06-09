#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "biio.h"

#include "recblocking_solver.h"
#include "recblocking_solver_cuda.h"

// "Usage: ``./sptrsv-double -d 0 -rhs 1 -lv -1 -forward/-backward -mtx A.mtx'' for Ax=b on device 0"
int main(int argc,  char ** argv)
{
    // report precision of floating-point
    // printf("---------------------------------------------------------------------------------------------\n");
    char  *precision;
    if (sizeof(VALUE_TYPE) == 4)
    {
        precision = (char *)"32-bit Single Precision";
    }
    else if (sizeof(VALUE_TYPE) == 8)
    {
        precision = (char *)"64-bit Double Precision";
    }
    else
    {
        // printf("Wrong precision. Program exit!\n");
        return 0;
    }

    // printf("PRECISION = %s\n", precision);
    // printf("Benchmark REPEAT = %i\n", BENCH_REPEAT);
    // printf("---------------------------------------------------------------------------------------------\n");

    int m, n, nnzA, isSymmetricA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;

    int nnzTR;
    int *csrRowPtrTR;
    int *csrColIdxTR;
    VALUE_TYPE *csrValTR;

    int device_id = 0;
    int rhs = 0;
    int lv = 0;
    int substitution;

    int argi = 1;

    // load device id
    char *devstr;
    if(argc > argi)
    {
        devstr = argv[argi];
        argi++;
    }

    if (strcmp(devstr, "-d") != 0) return 0;

    if(argc > argi)
    {
        device_id = atoi(argv[argi]);
        argi++;
    }
    // printf("device_id = %i\n", device_id);

    // load the number of right-hand-side
    char *rhsstr;
    if(argc > argi)
    {
        rhsstr = argv[argi];
        argi++;
    }

    if (strcmp(rhsstr, "-rhs") != 0) return 0;

    if(argc > argi)
    {
        rhs = atoi(argv[argi]);
        argi++;
    }
    // printf("rhs = %i\n", rhs);

    // load the number of recursive levels
    char *lvstr;
    if(argc > argi)
    {
        lvstr = argv[argi];
        argi++;
    }

    if (strcmp(lvstr, "-lv") != 0) return 0;

    if(argc > argi)
    {
        lv = atoi(argv[argi]);
        argi++;
    }

    // load substitution, forward or backward
    char *substitutionstr;
    if(argc > argi)
    {
        substitutionstr = argv[argi];
        argi++;
    }

    if (strcmp(substitutionstr, "-forward") == 0)
        substitution = SUBSTITUTION_FORWARD;
    else if (strcmp(substitutionstr, "-backward") == 0)
        substitution = SUBSTITUTION_BACKWARD;
    // printf("substitutionstr = %s\n", substitutionstr);
    // printf("substitution = %i\n", substitution);

    // load matrix file type, mtx, cscl, or cscu
    char *matstr;
    if(argc > argi)
    {
        matstr = argv[argi];
        argi++;
    }
    // printf("matstr = %s\n", matstr);

    // load matrix data from file
    char  *filename;
    if(argc > argi)
    {
        filename = argv[argi];
        argi++;
    }
    // printf("-------------- %s --------------\n", filename);

    srand(time(NULL));

    // load mtx data to the csr format
    // mmio_info(&m, &n, &nnzA, &isSymmetricA, filename);
    // csrRowPtrA = (int *)malloc((m+1) * sizeof(int));
    // csrColIdxA = (int *)malloc(nnzA * sizeof(int));
    // csrValA    = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE));
    // mmio_data(csrRowPtrA, csrColIdxA, csrValA, filename);
    // printf("input matrix A: ( %i, %i ) nnz = %i\n", m, n, nnzA);
    
    double *hA_values_tmp;
    read_Dmatrix(&m, &n, &nnzA, &csrRowPtrA, &csrColIdxA, &hA_values_tmp, &isSymmetricA, filename);
    csrValA = (VALUE_TYPE *)malloc(nnzA * sizeof(VALUE_TYPE));
    for (size_t i = 0; i < nnzA; i++)
    {
        csrValA[i] = (VALUE_TYPE)(hA_values_tmp[i]);
    }
    free(hA_values_tmp);
    
    if (m!=n)
    {
        printf("we need square matrix. Exit!\n");
        return 0;
    }

    // extract L and U with a unit diagonal of A
    csrRowPtrTR = (int *)malloc((m+1) * sizeof(int));
    csrColIdxTR = (int *)malloc((m+nnzA) * sizeof(int));
    csrValTR = (VALUE_TYPE *)malloc((m+nnzA) * sizeof(VALUE_TYPE));

    int nnz_pointer = 0;
    csrRowPtrTR[0] = 0;
    if (substitution == SUBSTITUTION_FORWARD)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
            {   
                if (csrColIdxA[j] < i)
                {
                    csrColIdxTR[nnz_pointer] = csrColIdxA[j];
                    csrValTR[nnz_pointer] = rand() % 10 + 1; //csrValA[j]; 
                    nnz_pointer++;
                }
            }
            // add dia nonzero
            csrColIdxTR[nnz_pointer] = i;
            csrValTR[nnz_pointer] = 1.0;
            nnz_pointer++;

            csrRowPtrTR[i+1] = nnz_pointer;
        }   
    }
    else
    {
        for (int i = 0; i < m; i++)
        {
            // add dia nonzero
            csrColIdxTR[nnz_pointer] = i;
            csrValTR[nnz_pointer] = 1.0;
            nnz_pointer++;
            
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i+1]; j++)
            {   
                if (csrColIdxA[j] > i)
                {
                    csrColIdxTR[nnz_pointer] = csrColIdxA[j];
                    csrValTR[nnz_pointer] = rand() % 10 + 1; //csrValA[j]; 
                    nnz_pointer++;
                }
            }
            csrRowPtrTR[i+1] = nnz_pointer;
        }
    }

    int nnz_tmp = csrRowPtrTR[m];
    nnzTR = nnz_tmp;

    csrColIdxTR = (int *)realloc(csrColIdxTR, sizeof(int) * nnzTR);
    csrValTR = (VALUE_TYPE *)realloc(csrValTR, sizeof(VALUE_TYPE) * nnzTR);
    
    free(csrColIdxA);
    free(csrValA);
    free(csrRowPtrA);
    
    // perm b and y (Ly=b and Ux=y)
    VALUE_TYPE *x_ref  =  (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n * rhs);
    VALUE_TYPE *x  =  (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * n * rhs);
    for (int i = 0; i < n * rhs; i++)
    {
        x_ref[i] = rand() % 10 + 1;
    }

    VALUE_TYPE *b  =  (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * m * rhs);
    memset(b, 0, sizeof(VALUE_TYPE) * m * rhs);
    for (int r = 0; r < rhs; r++)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = csrRowPtrTR[i]; j < csrRowPtrTR[i + 1]; j++)
            {
                b[r * m + i] += csrValTR[j] * x_ref[r * n + csrColIdxTR[j]];
            }
        }
    }

    // transpose CSR of U and L to CSC
    int *cscColPtrTR = (int *)malloc(sizeof(int) * (n + 1));
    cscColPtrTR[0] = 0;
    int *cscRowIdxTR = (int *)malloc(sizeof(int) * nnzTR);
    VALUE_TYPE *cscValTR = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * nnzTR);
    matrix_transposition(m, n, nnzTR,
                             csrRowPtrTR, csrColIdxTR, csrValTR,
                             cscRowIdxTR, cscColPtrTR, cscValTR);


    free(csrRowPtrTR);
    free(csrColIdxTR);
    free(csrValTR);
        
    if (lv == -1)
    {
        int li = 1;
        for (li = 1; li <= 100; li++)
        {
            if (m / pow(2, (li+1)) < (device_id == 0 ? 92160 : 58880)) // 92160 (4608x20) is titan rtx, 58880 (2944x20) is rtx 2080
                break;
        }
        lv = li;
    }
    
    int *d_cscColPtrTR;
    int *d_cscRowIdxTR;
    VALUE_TYPE *d_cscValTR;
    cudaMalloc((void **)&d_cscColPtrTR, (n + 1) * sizeof(int));
    cudaMalloc((void **)&d_cscRowIdxTR, nnzTR * sizeof(int));
    cudaMalloc((void **)&d_cscValTR, nnzTR * sizeof(VALUE_TYPE));
    
    cudaMemcpy(d_cscColPtrTR, cscColPtrTR, sizeof(int) * (n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscRowIdxTR, cscRowIdxTR, sizeof(int) * nnzTR, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cscValTR, cscValTR, sizeof(VALUE_TYPE) * nnzTR, cudaMemcpyHostToDevice);


    VALUE_TYPE *d_x;
    VALUE_TYPE *d_b;
    cudaMalloc((void **)&d_x, m * sizeof(VALUE_TYPE));
    cudaMalloc((void **)&d_b, m * sizeof(VALUE_TYPE));
    
    cudaMemcpy(d_x, x, sizeof(VALUE_TYPE) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(VALUE_TYPE) * m, cudaMemcpyHostToDevice);
    
    double cal_time = 0;
    double preprocess_time = 0;
    recblocking_solver_cuda(d_cscColPtrTR, d_cscRowIdxTR, d_cscValTR,
                            m, n, nnzTR, d_x, d_b, substitution, lv, &cal_time, &preprocess_time);
    cudaMemcpy(x, d_x, sizeof(VALUE_TYPE) * m, cudaMemcpyDeviceToHost);

    printf("%d,%d,%d,%d,%d,%d,%f,%f\n", sizeof(VALUE_TYPE), sizeof(VALUE_TYPE), sizeof(VALUE_TYPE), m, n, nnzTR, cal_time, (2 * nnzTR) / (cal_time * 1e6));
    
    // printf("computation usetime = %.3lf ms\n", cal_time);
    // printf("Performance = %.3lf gflops\n", (2 * nnzTR) / (cal_time * 1e6));

    cudaFree(d_cscColPtrTR);
    cudaFree(d_cscRowIdxTR);
    cudaFree(d_cscValTR);
    cudaFree(d_b);
    cudaFree(d_x);
    free(cscColPtrTR);
    free(cscRowIdxTR);
    free(cscValTR);
    free(b);
    
    
    // validate x
    int flag = 0;
    double accuracy = 1e-4;
    double ref = 0.0;
    double res = 0.0;

    for (int i = 0; i < n * rhs ; i++)
    {
        ref += abs(x_ref[i]);
        res += abs(x[i] - x_ref[i]);
    }
    // printf("\n");

    res = ref == 0 ? res : res / ref;
    
    if (res < accuracy && (res >= 0))
    {
        // printf("x check passed! |x-xref|/|xref| = %8.2e\n", res);
        flag = 1;
    }
    else{
        // printf(" x check _NOT_ passed! |x-xref|/|xref| = %8.2e\n", res);
    }

    free(x);
    free(x_ref);
    
    return 0;
}