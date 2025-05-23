#ifndef MATOPS_H
#define MATOPS_H

// lets define some matrix data structures
struct matrix
{
    int rows;
    int cols;
    double *data;
};

matrix *mat_mul(matrix *A, matrix *B);
matrix *mat_add(matrix *A, matrix *B);
matrix *mat_sub(matrix *A, matrix *B);
matrix *mat_inv(matrix *A);
matrix *mat_transpose(matrix *A);
void display_matrix(matrix *A);
#endif // MATOPS_H