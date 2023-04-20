#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "biio.h"
#include "findlevel.h"

int main(int argc, char **argv)
{
    int row;
    int col;
    int nnz;
    int *row_ptr;
    int *col_idx;
    double *val;
    int isSymmeticeR;
    char *filename = argv[1];

    read_Dmatrix(&row, &col, &nnz, &row_ptr, &col_idx, &val, &isSymmeticeR, filename);
    
    INFO_LOG("read over\n");

    // extract L with the unit-lower triangular sparsity structure of A
    int nnzL = 0;
    int *csrRowPtrL_tmp = (int *)malloc((row+1) * sizeof(int));
    int *csrColIdxL_tmp = (int *)malloc(nnz * sizeof(int));
    VALUE_TYPE *csrValL_tmp    = (VALUE_TYPE *)malloc(nnz * sizeof(VALUE_TYPE));

    int nnz_pointer = 0;
    csrRowPtrL_tmp[0] = 0;
    for (int i = 0; i < row; i++)
    {
        for (int j = row_ptr[i]; j < row_ptr[i+1]; j++)
        {
            if (col_idx[j] < i)
            {
                csrColIdxL_tmp[nnz_pointer] = col_idx[j];
                csrValL_tmp[nnz_pointer] = 1.0; //csrValA[j];
                nnz_pointer++;
            }
            else
            {
                break;
            }
        }

        csrColIdxL_tmp[nnz_pointer] = i;
        csrValL_tmp[nnz_pointer] = 1.0;
        nnz_pointer++;

        csrRowPtrL_tmp[i+1] = nnz_pointer;
    }

    nnzL = csrRowPtrL_tmp[row];
    INFO_LOG("A's unit-lower triangular L: ( %i, %i ) nnz = %i\n", row, col, nnzL);

    free(row_ptr);
    free(col_idx);
    free(val);

    int nlevel;
    int parallelism_min;
    int parallelism_avg;
    int parallelism_max;

    findlevel_csr(csrRowPtrL_tmp,
                  csrColIdxL_tmp,
                  csrValL_tmp,
                  row,
                  col,
                  nnzL,
                  &nlevel,
                  &parallelism_min,
                  &parallelism_avg,
                  &parallelism_max);

    printf("%s,%d,%d,%d,%d\n", filename, nlevel, parallelism_min, parallelism_avg, parallelism_max);

    return 0;
}