#ifndef BAGGING_UTILS
#define BAGGING_UTILS

#include <vector>
#include <armadillo>

#include "types.hpp"


namespace BaggingUtils {

typedef std::vector<std::vector<int>> std_matrix;
/*
    A struct to store the sampled data in. If each of the vectors
    contains more than one item, than it means that the class
    balance is being preserved, and data is separated into
    positive and negative data.
    * THE ORDER IS ALWAYS (1) NEGATIVE, (2) POSITIVE
*/
struct sampled_data {
    std::vector<Matrix> x_train;
    std::vector<Vector> y_train;
    std::vector<Matrix> x_valid;
    std::vector<Vector> y_valid;
};

/*
    Draws a random reservoir sample of size k
*/
std::vector<int> reservoir_sample ( const int, const std::vector<int>& );


std::vector<int> sampling_without_replacement ( const int, std::vector<int> );

/*
    Splits data into a training and a validation set. (no balance)
*/
sampled_data split_data ( const Matrix&, const double, const bool = true );

/*
    BFS
*/
int bfs ( std_matrix&, int, int );

/*
    Ford-Fulkerson algorithm for Maximum Network Flow
*/
int max_flow ( std_matrix&, int, int, std_matrix& );

}

#endif // BAGGING_UTILS
