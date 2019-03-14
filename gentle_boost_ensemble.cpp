#include "gentle_boost_ensemble.hpp"
#include <iostream>
/*!
    Custom exception for constructor failure handling
!*/
class unsupported_value_exception : public std::exception {
    virtual const char* what() const throw() {
        return "Variable _T has an unsupported value.";
    }
} myex;
/*!
    Destructor - resets all std::shared_ptr instances
!*/
GentleBoostEnsemble::~GentleBoostEnsemble ( ) {
    for (auto&& i : _base_learners) {
        i.reset();
    }
}

std::shared_ptr<GentleBoostEnsemble> GentleBoostEnsemble::get_ptr ( ) {
    return shared_from_this();
}

/*!
    Default constructor
!*/
GentleBoostEnsemble::GentleBoostEnsemble ( ) { }

/*!
    Constructor by training data
!*/
GentleBoostEnsemble::GentleBoostEnsemble ( const Matrix& x, const Vector& y ) {
    _x = x;
    _y = y;
    _x_weights = 1.0 / (double)x.n_rows * arma::ones<Vector>(y.n_elem);
    for(int i = 0; i < x.n_cols; ++i) {
        _max.push_back(x.col(i).max());
    }
}

/*!
    Copy constructor
!*/
GentleBoostEnsemble::GentleBoostEnsemble ( const GentleBoostEnsemble& other ) {
    _x = other._x;
    _y = other._y;
    _x_weights = other._x_weights;
    for(decltype(_base_learners.size()) i = 0; i < other._base_learners.size(); ++i) {
        _base_learners.push_back(std::make_shared<RegressionStump>(*other._base_learners[i]));
        _base_learners[i]->set_parent_ensemble(get_ptr());
    }
    _iteration = other._iteration;
    _weight = other._weight;
    for(auto&& i : other._max) {
        _max.push_back(i);
    }
}

void GentleBoostEnsemble::update_max ( ) {
    int attr = 0;
    for(auto&& m : _max) {
        m = _x.col(attr).max();
        attr++;
    }
}

/*
    Creates the base learners
*/
const bool GentleBoostEnsemble::create_base_learners ( ) {
    if(_T > 0) {
        for(int i = 0; i < _T; ++i) {
            _base_learners.push_back(std::make_shared<RegressionStump>());
            _base_learners[i]->set_parent_ensemble(get_ptr());
        }

        return true;
    }
    return false;
}

/*!
    Trains a regression stump (one step in gentle boost training)
!*/
void GentleBoostEnsemble::train_single ( ) {
    _base_learners[_iteration]->train();
    update_weights(_base_learners[_iteration]->compute_margins());
    normalize_weights();
}

void GentleBoostEnsemble::update_and_normalize_weights() {
    update_weights(_base_learners[_iteration]->compute_margins());
    normalize_weights();
}

void GentleBoostEnsemble::train_single_no_update ( ) {
    _base_learners[_iteration]->train();
}

/*!
    Trains the ensemble for 'n' iterations
!*/
void GentleBoostEnsemble::train ( const int& n ) {
    for(int i = 0; i < n-1; ++i) {
        train_single();
        _iteration++;
    }
}

/*
    Retrains the current weak learner. Used within collaboration, so that
    no weight updates/normalizations are performed at all.
*/
void GentleBoostEnsemble::retrain_current ( int howmany ) {
    // first compute the squared loss
    double squared_loss_min = accu(_x_weights % arma::pow((_base_learners[_iteration]->compute_margins()-1.0), 2));
    _base_learners[_iteration]->retrain_on_last(squared_loss_min, howmany);


}


/*!
    Performs full gentle boost training in _T iterations.
!*/
void GentleBoostEnsemble::train_full ( Matrix& x_tr, Vector& y_tr, Matrix& x_ts, Vector& y_ts ) {
    for(int i = 0; i < _T; ++i) {
        train_single();
        _iteration++;
        report_errors(x_tr, y_tr, x_ts, y_ts);

    }
}

/*!
    Outputs the real-valued predictions on the trainig set _x
!*/
Vector GentleBoostEnsemble::predict_real ( ) const {
    Vector f = arma::zeros<Vector>(_y.n_elem);
    int iter = 0;
    while(iter < _iteration) {
        f += _base_learners[iter]->predict();
        iter++;
    }
    return f;
}

/*!
    Outputs the real-valued predictions on a test set x
!*/
Vector GentleBoostEnsemble::predict_real ( const Matrix& x ) const {
    Vector f = arma::zeros<Vector>(x.n_rows);
    int iter = 0;
    while(iter < _iteration) {
        f += _base_learners[iter]->predict(x);
        iter++;
    }
    return f;
}

/*!
    Outputs the real predictions on the trainig set _x
!*/
Vector GentleBoostEnsemble::predict_discrete ( ) const {
    Vector f = arma::zeros<Vector>(_y.n_elem);
    int iter = 0;
    while(iter < _iteration) {
        f += _base_learners[iter]->predict();
        iter++;
    }
    return arma::sign(f);
}

/*!
    Outputs the discrete predictions on a test set x (-1, +1)
!*/
Vector GentleBoostEnsemble::predict_discrete ( const Matrix& x ) const {
    Vector f = arma::zeros<Vector>(x.n_rows);
    int iter = 0;
    while(iter < _iteration) {
        f += _base_learners[iter]->predict(x);
        iter++;
    }

    return arma::sign(f);
}

/*!
    Computes the margins on the training set _x
!*/
Vector GentleBoostEnsemble::compute_margins ( ) const {
    return _y % (predict_real());
}

/*!
    Computes the margins of a given set x and outputs y
!*/
Vector GentleBoostEnsemble::compute_margins ( const Matrix& x, const Vector& y ) const {
    return y % (predict_real(x));
}

double GentleBoostEnsemble::compute_error_rate ( const Matrix& x, const Vector& y) const {
    Vector f = predict_discrete(x);
    int misclassified = 0;
    for(int i = 0; i < f.n_elem; ++i)
        if(f(i) != y(i)) misclassified++;

    return (double) misclassified / (double) y.n_elem;
}

void GentleBoostEnsemble::report_errors ( Matrix& x_train, Vector& y_train, Matrix& x_test, Vector& y_test ) const {
    double tr_err = compute_error_rate(x_train, y_train);
    double ts_err = compute_error_rate(x_test, y_test);
    std::cout << tr_err << "," << ts_err << std::endl;
}
