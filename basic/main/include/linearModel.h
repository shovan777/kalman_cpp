#ifndef LINEARMODEL_H
#define LINEARMODEL_H
#include "matops.h"

class LinearModel {
public:
    // Constructor: assigns parameter pointers to member variables.
    LinearModel(matrix *state_vec, matrix *transition_matrix, matrix *input_matrix);

    // Set a new transition matrix.
    void setTransitionMatrix(matrix *transition_matrix);

    // Get the current transition matrix.
    matrix *getTransitionMatrix();

    // Update the state using the input vector.
    void updateState(matrix *input_vec);

    // Get the current state vector.
    matrix *getStateVector() { return state_vec; }

    // set the state vector
    void setStateVector(matrix *state_vec);
    // Destructor: free the memory allocated for the matrices.
    ~LinearModel();

private:
    matrix *state_vec;
    matrix *transition_matrix;
    matrix *input_matrix;
};
#endif // LINEARMODEL_H