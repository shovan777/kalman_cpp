// create a class that will handle the linear motion of the robot
#include "include/linearModel.h"
#include <iostream>
#include <cmath>

// Implementation of LinearMotion class methods

LinearModel::LinearModel(matrix *state_vec, matrix *transition_matrix, matrix *input_matrix)
{
    this->state_vec = state_vec;
    this->transition_matrix = transition_matrix;
    this->input_matrix = input_matrix;
}

void LinearModel::setTransitionMatrix(matrix *transition_matrix)
{
    this->transition_matrix = transition_matrix;
}

void LinearModel::setStateVector(matrix *state_vec)
{
    this->state_vec->data = state_vec->data;
}

matrix *LinearModel::getTransitionMatrix()
{
    return this->transition_matrix;
}

void LinearModel::updateState(matrix *input_vec)
{
    // update the state vector
    this->state_vec = mat_mul(this->transition_matrix, state_vec);
    matrix *temp = mat_mul(this->input_matrix, input_vec);
    this->state_vec = mat_add(temp, state_vec);
    free(temp->data);
}

// Destructor
LinearModel::~LinearModel()
{
    free(state_vec->data);
    free(transition_matrix->data);
    free(input_matrix->data);
}