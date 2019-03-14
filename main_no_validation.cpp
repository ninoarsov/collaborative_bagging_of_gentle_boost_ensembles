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

    if(argc != 9) {
        std::cerr << "Wrong arguments. The correct arguments format is:" <<
            "\t [#GB ensembles] [#data fraction] [#iterations] [#validation fraction] [#collaboration probability] [#instances to exchange] [#training set] [#test set]"
            << std::endl;
        exit(1);
    }

    // read the data paths
    std::string training_path = argv[7];
    std::string test_path = argv[8];

    // parse the cmd line args into options
    options opts;
    opts.n_ensembles = std::atof(argv[1]);
    opts.bagging_data_frac = std::atof(argv[2]);
    opts.bagging_ensembles_frac = 0.5;
    opts.T = std::atoi(argv[3]);
    opts.validation_frac = std::atof(argv[4]);
    opts.p_c = std::atof(argv[5]);
    opts.instances_to_exchange = std::atoi(argv[6]);

    // set the static number of iterations
    GentleBoostEnsemble::_T = opts.T;

    // read the training and test data
    Matrix training_data = read_matrix_from_file(training_path);
    Matrix training_x = get_instances(training_data);
    Vector training_y = get_class_labels(training_data);
    Matrix test_data = read_matrix_from_file(test_path);
    Matrix test_x = get_instances(test_data);
    Vector test_y = get_class_labels(test_data);

    opts.train_data = &training_x;
    opts.train_labels = &training_y;
    opts.test_data = &test_x;
    opts.test_labels = &test_y;

    // create a bagged boosting ensemble
    auto bbe =
    std::make_shared<BaggedBoostingEnsemble>(opts);

    // distribute the data
    bbe->distribute_training_data(training_data);

    auto bbe_no_clb = std::make_shared<BaggedBoostingEnsemble>(opts);
    bbe_no_clb->distribute_training_data(training_data);
    for(int i = 0; i < opts.n_ensembles; i++) {
        bbe_no_clb->get_ensembles()[i]->set_x(bbe->get_ensembles()[i]->get_x());
        bbe_no_clb->get_ensembles()[i]->set_y(bbe->get_ensembles()[i]->get_y());
        bbe_no_clb->get_ensembles()[i]->set_x_weights(bbe->get_ensembles()[i]->get_x_weights());
        bbe_no_clb->set_validation_data(bbe->get_validation_data());
        bbe_no_clb->set_validation_labels(bbe->get_validation_labels());
    }

    // train (uses collaboration by default)
    bbe->train(true, 5, false);

    std::cout << "<<<< No collaboration >>>>" << std::endl;
    bbe_no_clb->train(false, SQUARED_ERROR_MINIMIZATION, false);

    return 0;
}
