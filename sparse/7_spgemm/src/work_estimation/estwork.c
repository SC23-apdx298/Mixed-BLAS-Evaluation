#include "biio.h"

int main(int argc, char **argv)
{
    int A_num_rows;
    int A_num_cols;
    int A_nnz;
    int C_nnz;
    int *hA_csrOffsets;
    int *hA_columns;
    double *hA_values;
    int isSymmetricA;
    char *filename = argv[1];

    read_Dmatrix(&A_num_rows, &A_num_cols, &A_nnz, &hA_csrOffsets, &hA_columns, &hA_values, &isSymmetricA, filename);

    if (A_num_rows != A_num_cols)
    {
        free(hA_csrOffsets);
        free(hA_columns);
        free(hA_values);
        printf("mat must be square, exit!\n");
        return -1;
    }

    int B_num_rows = A_num_rows;
    int B_num_cols = A_num_cols;
    int B_nnz = A_nnz;
    int *hB_csrOffsets = hA_csrOffsets;
    int *hB_columns = hA_columns;
    double *hB_values = hA_values;

    u_int64_t nnzCub = 0;

    for (int i = 0; i < A_nnz; i++)
    {
        int rowB = hA_columns[i];
        nnzCub += hA_csrOffsets[rowB + 1] - hB_csrOffsets[rowB];
    }

    printf("%lu\n",nnzCub);

    free(hA_csrOffsets);
    free(hA_columns);
    free(hA_values);
    return 0;
}