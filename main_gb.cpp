#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>

#define ARMA_NO_DEBUG
#define ARMA_DONT_PRINT_ERRORS

#include "bagged_boosting_ensemble.hpp"

Matrix read_matrix_from_file ( std::string path ) {

    Matrix result;
    result.load(path);

    return result;
}

Vector get_class_labels ( const Matrix& data ) {
    Vector y = data.col(data.n_cols - 1);
    for(arma::uword i = 0; i < y.n_elem; ++i)
        if(y(i) == 0.0) y(i) = -1.0;
    return y;
}

Matrix get_instances ( const Matrix& data ) {
    Matrix x = data.submat(0, 0, data.n_rows - 1, data.n_cols - 2);
    return x;
}

int GentleBoostEnsemble::_T = -1;
int SQUARED_ERROR_MINIMIZATION = 0;
int DIVERSITY_MAXIMIZATION = 1;


int main ( int argc, char* argv[] ) {

    if(argc != 4) {
        std::cerr << "Wrong arguments. The correct arguments format is:" <<
            "\t [#iterations] [#training set] [#test set]"
            << std::endl;
        exit(1);
    }

    // read the data paths
    std::string training_path = argv[2];
    std::string test_path = argv[3];

    // set the static number of iterations
    GentleBoostEnsemble::_T = std::atoi(argv[1]);

    // read the training and test data
    Matrix training_data = read_matrix_from_file(training_path);
    Matrix training_x = get_instances(training_data);
    Vector training_y = get_class_labels(training_data);
    Matrix test_data = read_matrix_from_file(test_path);
    Matrix test_x = get_instances(test_data);
    Vector test_y = get_class_labels(test_data);


    // GB
    auto gbe = std::make_shared<GentleBoostEnsemble>(training_x, training_y);
    gbe->create_base_learners();
    gbe->train_full(training_x, training_y, test_x, test_y);

    return 0;
}
