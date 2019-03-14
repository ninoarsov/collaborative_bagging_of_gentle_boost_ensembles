#include "types.hpp"


Vector arma::sigmoid(const Vector& x) {
    return 1.0 / (1.0 + exp(-x));
}

RowVector arma::sigmoid(const RowVector& x) {
    return 1.0 / (1.0 + exp(-x));
}

Matrix arma::sigmoid(const Matrix& x) {
    return 1.0 / (1.0 + exp(-x));
}
