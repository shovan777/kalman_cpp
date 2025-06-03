#include <stdio.h>
#include <stdlib.h>
#include "include/matops.h"
#include <iostream>
#include <omp.h>

// lets define some matrix operations
// return just the pointer to the matrix
matrix *mat_mul(matrix *A, matrix *B)
{
    int m = A->rows;
    int n = A->cols;
    int p = B->cols;

    // check if the matrix multiplication is possible
    if (A->cols != B->rows)
    {
        printf("Matrix multiplication not possible\n");
        return NULL;
    }

    matrix *C = new matrix;
    C->rows = m;
    C->cols = p;
    C->data = (long double*)malloc(C->rows * C->cols * sizeof(long double));

    // #pragma omp parallel num_threads(4)
    // {
        // #pragma omp for
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < p; j++)
            {
                int c_idx = i * p + j;
                C->data[c_idx] = 0.0;
                for (int k = 0; k < n; k++)
                {
                    C->data[c_idx] += (A->data[i * n + k] * B->data[k * p + j]);
                }
            }
        }
    // }

    return C;
}

matrix *mat_add(matrix *A, matrix *B)
{
    matrix *C = new matrix;
    int m = A->rows;
    int n = A->cols;
    C->rows = m;
    C->cols = n;
    C->data = (long double *)malloc(C->rows * C->cols * sizeof(long double));
    for (int i = 0; i < C->rows; i++)
    {
        for (int j = 0; j < C->cols; j++)
        {
            C->data[i * C->cols + j] = A->data[i * A->cols + j] + B->data[i * B->cols + j];
        }
    }
    return C;
}

matrix *mat_sub(matrix *A, matrix *B)
{
    matrix *C = new matrix;
    int m = A->rows;
    int n = A->cols;
    C->rows = m;
    C->cols = n;
    C->data = (long double *)malloc(C->rows * C->cols * sizeof(long double));
    for (int i = 0; i < C->rows; i++)
    {
        for (int j = 0; j < C->cols; j++)
        {
            C->data[i * C->cols + j] = A->data[i * A->cols + j] - B->data[i * B->cols + j];
        }
    }
    return C;
}

matrix *mat_inv(matrix *A)
{
    //  lets use gauss jordan method and assume that mat A is invertible
    //  i.e A is also square
    // steps
    // 1: begin by creating a n by 2n matrix where n is dim(A)
    int num_rows = A->rows;
    matrix *identity_mat = new matrix;
    identity_mat->rows = num_rows;
    identity_mat->cols = num_rows;

    // only invert square matrices else raise error
    if (A->cols != num_rows)
    {
        // print an error
        printf("Only square matrix has an inverse");

        // throw an error
        return NULL;
    }
    identity_mat->data = (long double *)malloc(num_rows * num_rows * sizeof(long double));

    // start with an identity matrix
    for (int i = 0; i < num_rows; i++)
    {
        for (int j = 0; j < num_rows; j++)
        {
            int idx = i * num_rows + j;
            if (i == j)
            {
                identity_mat->data[idx] = 1.0;
            }
            else
            {
                identity_mat->data[idx] = 0.0;
            }
        }
    }
    // 2: start at a(1,1) diagonal element
    // #pragma omp parallel num_threads(2)
    // {
        // #pragma parallel for shared(A, identity_mat)
    for (int row = 0; row < num_rows; row++)
    {
        int pivot_row = row * num_rows;
        // get the diagonal element as pivot
        long double pivot = A->data[pivot_row + row];

        // if pivot is zero, matrix is not invertible
        if (pivot == 0.0)
        {
            printf("Matrix is not invertible %f\n", pivot);
            // return NULL;
        }
        // pivot = 0.0;
        // printf("div by zero %f",8/pivot);

        // divide the row by the pivot
        for (int col = 0; col < num_rows; col++)
        {
            A->data[row * num_rows + col] /= pivot;
            identity_mat->data[row * num_rows + col] /= pivot;
        }

        // eliminate all other elements in that [row][row] to zero
        for (int j = 0; j < num_rows; j++)
        {
            if (row == j)
            {
                continue;
            }
            int cur_row = j * num_rows;
            long double factor = A->data[cur_row + row];
            // every other element in that row is subtractor by factor times pivot row
            for (int col = 0; col < num_rows; col++)
            {
                A->data[cur_row + col] -= factor * A->data[pivot_row + col];
                identity_mat->data[cur_row + col] -= factor * identity_mat->data[pivot_row + col];
            }
        }
    }
// }
    return identity_mat;
}

matrix *mat_transpose(matrix *A)
{
    matrix *B = new matrix;
    B->rows = A->cols;
    B->cols = A->rows;
    B->data = (long double *)malloc(B->rows * B->cols * sizeof(long double));
    for (int i = 0; i < A->rows; i++)
    {
        for (int j = 0; j < A->cols; j++)
        {
            B->data[j * A->rows + i] = A->data[i * A->cols + j];
        }
    }
    return B;
}

void display_matrix(matrix *A)
{
    std::cout << "Matrix: " << std::endl;
    std::cout << "Rows: " << A->rows << ", Cols: " << A->cols << std::endl;
    std::cout << "Data: " << std::endl;
    for (int i = 0; i < A->rows; i++)
    {
        for (int j = 0; j < A->cols; j++)
        {
            std::cout << A->data[i * A->cols + j] << " ";
        }
        std::cout << std::endl;
    }
}