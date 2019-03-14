#ifndef TYPES_
#define TYPES_

#include <armadillo>

typedef arma::mat Matrix;
typedef arma::vec Vector;
typedef arma::rowvec RowVector;
typedef arma::umat UMatrix;
typedef arma::uvec UVector;

namespace arma {

Vector sigmoid(const Vector&);
RowVector sigmoid(const RowVector&);
Matrix sigmoid(const Matrix&);

}

#endif
