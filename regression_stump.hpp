#ifndef REGRESSION_STUMP
#define REGRESSION_STUMP

/* A regression stump weak classifier as a base learner within Gentle Boost */

#include <memory>
#include <cmath>
#include <type_traits>
#include <limits>

#include <armadillo>

#include "types.hpp"


/* Forward declaration */
class GentleBoostEnsemble;

class RegressionStump {

private:
    // the parent ensemble of this regression stump
    std::shared_ptr<GentleBoostEnsemble> _parent_ensemble;
    // regression stump parameters
    double _a;
    double _b;
    int _dimension;
    double _threshold;

public:

    // constructors
    RegressionStump ( );
    RegressionStump ( const GentleBoostEnsemble& );
    RegressionStump ( const RegressionStump& );
    ~RegressionStump ( );

    // setters
    inline void set_parent_ensemble ( std::shared_ptr<GentleBoostEnsemble> parent_ensemble ) { _parent_ensemble = parent_ensemble; }
    inline void set_a ( const double& a ) { _a = a; }
    inline void set_b ( const double& b ) { _b = b; }
    inline void set_dimension ( const int& dimension ) { _dimension = dimension; }
    inline void set_threshold ( const double& threshold ) { _threshold = threshold; }

    // getters (defined here for autoMatrixic in-lining)
    inline const std::shared_ptr<GentleBoostEnsemble>& get_parent_ensemble ( ) const { return _parent_ensemble; }
    inline const double& get_a ( ) const { return _a; }
    inline const double& get_b ( ) const { return _b; }
    inline const int& get_dimension ( ) const { return _dimension; }
    inline const double& get_threshold ( ) const { return _threshold; }

    // operators
    RegressionStump& operator= ( const RegressionStump& );

    // training functionalities
    void train ( );
    void retrain_on_last ( double , int);
    void retrain ( );

    // prediction functionalities
    Vector predict ( ) const;
    Vector predict ( const Matrix& ) const;
    double compute_margin ( const Matrix&, const double ) const;
    Vector compute_margins ( ) const;
    Vector compute_margins ( const Matrix&, const Vector& ) const;

};

#endif //REGRESSION_STUMP
