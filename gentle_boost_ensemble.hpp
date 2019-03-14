#ifndef GENTLE_BOOST_ENSEMBLE
#define GENTLE_BOOST_ENSEMBLE

/* Implements a Gentle Boost classification ensemble */

#include <type_traits>
#include <memory>
#include <limits>
#include <vector>
#include <exception>

#include <armadillo>

#include "types.hpp"
#include "regression_stump.hpp"


/*!
    Implements a Gentle Boost ensemble (Friedmann et al.).
    It inherits std::enable_shared_from_this because multiple RegressionStump
    objects will contain a std::shared_ptr to this object. Therefore a
    shared_from_this() shared_ptr is passed to avoid double deletion.
!*/
class GentleBoostEnsemble: public std::enable_shared_from_this<GentleBoostEnsemble> {

private:
    Matrix _x;
    Vector _y;
    Vector _x_weights;
    std::vector<std::shared_ptr<RegressionStump>> _base_learners;
    int _iteration = 0;
    double _weight = 1.0;

    std::shared_ptr<GentleBoostEnsemble> get_ptr ( );

public:
    // max number of weak learners (max number of boosting iterations)
    static int _T;
    // misc
    std::vector<double> _max;

    // constructors
    GentleBoostEnsemble ( );
    GentleBoostEnsemble ( const Matrix&, const Vector& );
    GentleBoostEnsemble ( const GentleBoostEnsemble& );
    ~GentleBoostEnsemble ( );

    // misc
    const bool create_base_learners ( );


    // setters
    void set_x ( const Matrix& x ) { _x = x; }
    void set_y ( const Vector& y ) { _y = y; }
    void set_x_weights ( const Vector& x_weights ) { _x_weights = x_weights; }
    void set_base_learners ( const std::vector<std::shared_ptr<RegressionStump>> base_learners) { _base_learners = base_learners; }
    void set_iteration ( const int& iteration ) { _iteration = iteration; }
    void set_weight ( const double& weight ) { _weight = weight; }

    //getters
    const Matrix& get_x ( ) const { return _x; }
    Matrix get_x_copy ( ) const { return _x; }
    Matrix& get_x_nonconst ( ) { return _x; }
    const Vector& get_y ( ) const  { return _y; }
    Matrix get_y_copy ( ) const { return _y; }
    Vector& get_y_nonconst ( ) { return _y; }
    const Vector& get_x_weights ( ) const { return _x_weights; }
    Vector get_x_weights_copy ( ) const { return _x_weights; }
    Vector& get_x_weights_nonconst ( ) { return _x_weights; }
    const std::vector<std::shared_ptr<RegressionStump>>& get_base_learners ( ) const { return _base_learners; }
    const int& get_iteration ( ) const { return _iteration; }
    const double& get_weight ( ) const { return _weight; }

    // training functionalities
    friend void RegressionStump::train ( );
    friend void RegressionStump::retrain_on_last ( double, int );
    void train_single ( );
    void train_single_no_update ( );
    void train ( const int& );
    void retrain_current ( int );
    void train_full ( Matrix&, Vector&, Matrix&, Vector& );

    inline void update_weights ( const Vector& );
    void update_and_normalize_weights();
    inline void reset_weights_uniform ( );
    inline void normalize_weights ( );
    void update_max ( );

    //prediction functionalities
    friend inline Vector RegressionStump::predict ( ) const;
    friend inline Vector RegressionStump::predict ( const Matrix& ) const;
    friend inline Vector RegressionStump::compute_margins ( ) const;
    friend inline Vector RegressionStump::compute_margins ( const Matrix&, const Vector& ) const;
    Vector predict_real ( ) const;
    Vector predict_real ( const Matrix& ) const;
    Vector predict_discrete ( ) const;
    Vector predict_discrete ( const Matrix& ) const;
    Vector compute_margins ( ) const;
    Vector compute_margins ( const Matrix&, const Vector& ) const;
    double compute_error_rate ( const Matrix&, const Vector& ) const;
    void report_errors ( Matrix&, Vector&, Matrix&, Vector& ) const;

};


/*!
    Normalize the weights such that they all sum up to one
!*/
inline void GentleBoostEnsemble::normalize_weights ( ) {
    _x_weights /= sum(_x_weights);
}

/*!
    Updates the weights by the negative of the margins
!*/
inline void GentleBoostEnsemble::update_weights ( const Vector& arg ) {
    Vector exponents = arma::exp(-arg);
    _x_weights = _x_weights % exponents;
}

/*!
    Resets the weights so that they form a uniform probability distribution
!*/
inline void GentleBoostEnsemble::reset_weights_uniform ( ) {
    _x_weights = 1.0/ (double)_x_weights.n_elem * arma::ones<Vector>(_x_weights.n_elem);
}


#endif //GENTLE_BOOST_ENSEMBLE
