#include "regression_stump.hpp"
#include "gentle_boost_ensemble.hpp"

#include <iostream>


/*!
    Destructor
!*/
RegressionStump::~RegressionStump ( ) {
    //  _parent_ensemble.reset();
}

/*!
    Default constructor
!*/
RegressionStump::RegressionStump ( ) { }

/*!
    Constructor by parent gentle boost ensemble
!*/
RegressionStump::RegressionStump ( const GentleBoostEnsemble& parent_ensemble ) {
    _parent_ensemble = std::make_shared<GentleBoostEnsemble>(parent_ensemble);
}

/*!
    Copy constructor
!*/
RegressionStump::RegressionStump ( const RegressionStump& other ) :
    _a(other._a), _b(other._b), _dimension(other._dimension), _threshold(other._threshold) {
    _parent_ensemble = std::make_shared<GentleBoostEnsemble>(*other._parent_ensemble);
}

/*!
    Assignment operator overloading
!*/
RegressionStump& RegressionStump::operator= ( const RegressionStump& rhs ) {
    if(this != &rhs) {
        _parent_ensemble = std::make_shared<GentleBoostEnsemble>(*rhs._parent_ensemble);
        _a = rhs._a;
        _b = rhs._b;
        _dimension = rhs._dimension;
        _threshold = rhs._threshold;
    }
    return *this;
}

/*!
    Finds the optimal dimension and optimal threshold for the regression stump.
    Sets the values of _a, _b, _dimension, _threshold.
!*/
void RegressionStump::train ( ) {
    double min_wsse = std::numeric_limits<double>::infinity();
    double wy_dot =arma::dot(_parent_ensemble->_x_weights, _parent_ensemble->_y);
    Vector wy = _parent_ensemble->_x_weights % (_parent_ensemble->_y);
    for(arma::uword j = 0; j < _parent_ensemble->_x.n_cols; ++j) {
        for(arma::uword i = 0; i < _parent_ensemble->_x.n_rows; ++i) {
            if(_parent_ensemble->_x(i, j) == _parent_ensemble->_max[j])
                continue;
            double thr = _parent_ensemble->_x(i, j);
            Vector indicator = arma::zeros<Vector>(_parent_ensemble->_x.n_rows);
            for(arma::uword k = 0; k < _parent_ensemble->_x.n_rows; ++k) {
                if(_parent_ensemble->_x(k, j) > thr) indicator(k) = 1.0;
            }
            double wyi_dot = arma::dot(wy, indicator);
            double wi_dot = arma::dot(_parent_ensemble->_x_weights, indicator);
            double a = (wyi_dot - wi_dot * wy_dot) / (wi_dot - wi_dot * wi_dot);
            double b = (wy_dot - wyi_dot) / (1.0 - wi_dot);

            Vector f = a * indicator + b;
            Vector sq_err = pow(_parent_ensemble->_y - f, 2.0);
            double wsse = arma::dot(_parent_ensemble->_x_weights, sq_err);
            if (wsse < min_wsse) {
                min_wsse = wsse;
                _a = a;
                _b = b;
                _dimension = j;
                _threshold = thr;
            }
        }
    }

    // std::cout << _a << " " << _b << " " << _dimension << " " << _threshold << std::endl;
}

void RegressionStump::retrain_on_last ( double min_wsse, int howmany ) {
    double wy_dot =arma::dot(_parent_ensemble->_x_weights, _parent_ensemble->_y);
    Vector wy = _parent_ensemble->_x_weights % (_parent_ensemble->_y);
    for(arma::uword j = 0; j < _parent_ensemble->_x.n_cols; ++j) {
        for(arma::uword i = _parent_ensemble->_x.n_rows-howmany; i < _parent_ensemble->_x.n_rows; ++i) {
            if(_parent_ensemble->_x(i, j) == _parent_ensemble->_max[j])
                continue;
            double thr = _parent_ensemble->_x(i, j);
            Vector indicator = arma::zeros<Vector>(_parent_ensemble->_x.n_rows);
            for(arma::uword k = 0; k < _parent_ensemble->_x.n_rows; ++k) {
                if(_parent_ensemble->_x(k, j) > thr) indicator(k) = 1.0;
            }
            double wyi_dot = arma::dot(wy, indicator);
            double wi_dot = arma::dot(_parent_ensemble->_x_weights, indicator);
            double a = (wyi_dot - wi_dot * wy_dot) / (wi_dot - wi_dot * wi_dot);
            double b = (wy_dot - wyi_dot) / (1.0 - wi_dot);

            Vector f = a * indicator + b;
            Vector sq_err = pow(_parent_ensemble->_y - f, 2.0);
            double wsse = arma::dot(_parent_ensemble->_x_weights, sq_err);
            if (wsse < min_wsse) {
                min_wsse = wsse;
                _a = a;
                _b = b;
                _dimension = j;
                _threshold = thr;
            }
        }
    }
}

/*!
    Make predictions for the training set
!*/
Vector RegressionStump::predict ( ) const {
    Vector indicator = arma::zeros<Vector>(_parent_ensemble->_x.n_rows);
    for(arma::uword i = 0; i < indicator.n_elem; ++i) {
        if(_parent_ensemble->_x(i, _dimension) > _threshold) indicator(i) = 1.0;
    }
    Vector f = _a * indicator + _b;
    return f;
}

/*!
    Make predictions for a given test set x
!*/
Vector RegressionStump::predict ( const Matrix& x ) const {
    Vector indicator = arma::zeros<Vector>(x.n_rows);
    for(arma::uword i = 0; i < x.n_rows; ++i) {
        if(x(i, _dimension) > _threshold) indicator(i) = 1.0;
    }
    Vector f = _a * indicator + _b;
    return f;
}

/*!
    Compute the margins of the training set
!*/
Vector RegressionStump::compute_margins ( ) const {
    return _parent_ensemble->_y % (predict());
}

/*!
    Compute the margins of each instance for a given set x and outputs y
!*/
Vector RegressionStump::compute_margins ( const Matrix& x, const Vector& y ) const {
    return y % (predict(x));
}

/*
    Compute the margin of an instance
*/
double RegressionStump::compute_margin ( const Matrix& x, const double y ) const {
    Vector f = predict(x);
    return y * f(0);
}
