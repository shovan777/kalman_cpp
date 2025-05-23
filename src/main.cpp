#include <iostream>
#include <fstream>
#include "include/matops.h"
#include "include/linearModel.h"    // was "include/linearMotion.h"

// set data path
#define DATA_PATH "/home/shovan/ohod_ws/kalman_cpp/data"

// test a linear hst model to predict the transformer RUL
// remaining useful life
int main()
{
    double delta_t = 0.1;
    double meas_rul = 0.0;
    // Linear model for predicting transformer RUL
    // state vector: [rul]
    // transition matrix: [1]
    // input matrix: [-1]
    // input vector: [delta_t]
    // measurement matrix: [1]
    // measurement vector: [rul]

    LinearModel lm = LinearModel(
        // motion starts at rest
        new matrix{1, 1, new double[1]{0.0}},
        new matrix{1, 1, new double[1]{1.0}},
        new matrix{1, 1, new double[1]{-1.0}});

    // initialize
    // state covariance matrix
    matrix *stateCOVMat = new matrix{1, 1, new double[1]{0.01}};
    // process noise covariance matrix
    matrix *processCOVMat = new matrix{1, 1, new double[1]{0.01}};
    // measurement noise covariance matrix
    matrix *measurementCOVMat = new matrix{1, 1, new double[1]{0.01}};
    // kalman gain matrix
    matrix *kalmanGainMat = new matrix{1, 1, new double[1]{0.0}};
    // measurement matrix
    matrix *measurementMat = new matrix{1, 1, new double[1]{1.0}};

    // load data from csv file
    // read the data from the csv file
    std::string inputPath = DATA_PATH "/rul_data.csv";
    std::string outputPath = DATA_PATH "/rul_output.csv";
    printf("Reading data from %s\n", inputPath.c_str());

    // open output file in write mode
    std::ofstream outputFile(outputPath);
    // check if file is opened
    if (!outputFile.is_open())
    {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }

    std::ifstream file(inputPath);
    // check if file is found
    if (!file.is_open())
    {
        std::cerr << "Error opening file" << std::endl;
        return 1;
    }
    std::string line;
    std::getline(file, line); // skip the header line

    // lets generate a sequence of input vectors
    int i = 0;
    while (!file.eof() && i < 10)
    {
        // read the next line
        std::getline(file, line);
        // print the data in the line
        std::cout << "Line: " << line << std::endl;

        // parse the line to get the measurement
        // split the line by comma
        // the first token is the delta_t
        // the second token is the meas_rul
        std::string delimiter = ",";
        size_t pos = 0;
        std::string token;
        pos = line.find(delimiter);
        token = line.substr(0, pos);
        delta_t = std::stod(token);
        line.erase(0, pos + delimiter.length());
        // the second token is the meas_rul
        pos = line.find(delimiter);
        token = line.substr(0, pos);
        meas_rul = std::stod(token);
        line.erase(0, pos + delimiter.length());
        // print the data in double without truncation
        std::cout << "delta_t: " << delta_t << ", meas_rul: " << meas_rul << std::endl;
        // prediction step
        matrix *input_vec = new matrix{1, 1, new double[1]{delta_t}};

        //  prediction step: x = Fx + Bu
        lm.updateState(input_vec);

        // P = FPF^T + Q, where Q is process noise
        stateCOVMat = mat_mul(lm.getTransitionMatrix(), mat_mul(stateCOVMat, mat_transpose(lm.getTransitionMatrix())));
        stateCOVMat = mat_add(stateCOVMat, processCOVMat);

        // K = PH^T(HPH^T + R)^-1, where R is measurement noise
        matrix *temp = mat_mul(mat_mul(measurementMat, stateCOVMat), mat_transpose(measurementMat));
        temp = mat_add(temp, measurementCOVMat);
        kalmanGainMat = mat_mul(mat_mul(stateCOVMat, mat_transpose(measurementMat)), mat_inv(temp));

        // innovation/error: y_err = z - Hx, where measurement vector z is supplied externally. 
        matrix *y_err = mat_sub(new matrix{1, 1, new double[1]{meas_rul}}, mat_mul(measurementMat, lm.getStateVector()));

        // correction step: x = x + K*y_err
        lm.setStateVector(mat_add(lm.getStateVector(), mat_mul(kalmanGainMat, y_err)));

        // P = (I - KH)P
        stateCOVMat = mat_mul(mat_sub(new matrix{1, 1, new double[1]{1.0}}, mat_mul(kalmanGainMat, measurementMat)), stateCOVMat);

        printf("State vector after correction: %f\n", lm.getStateVector()->data[0]);

        // write the output to the file
        outputFile << lm.getStateVector()->data[0] << std::endl;

        free(input_vec->data);
        free(input_vec);
        free(y_err->data);
        free(y_err);
        i++;
    }
    // close the file
    file.close();
    outputFile.close();
    printf("Program 1d finished successfully\n");
    return 0;
}