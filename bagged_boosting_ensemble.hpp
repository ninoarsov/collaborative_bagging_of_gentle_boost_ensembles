#ifndef BAGGED_BOOSTING_ENSEMBLE
#define BAGGED_BOOSTING_ENSEMBLE

#include <memory>
#include <exception>
#include <vector>
#include <limits>

#include <armadillo>

#include "types.hpp"
#include "bagging_utils.hpp"
#include "gentle_boost_ensemble.hpp"

extern int SQUARED_ERROR_MINIMIZATION;
extern int DIVERSITY_MAXIMIZATION;

/*
    Helper functions
*/
void remove_row ( Matrix&, const int );
void remove_coeff ( Vector&, const int );
void append_row ( Matrix&, const Matrix& );
void append_coeff ( Vector&, double );
bool is_equal ( Matrix&, Matrix& );
/*
    Diversity function
*/
double diversity ( const std::vector<double>&, std::vector<double>&, void* );

struct options {
    int n_ensembles;
    double validation_frac;
    double bagging_data_frac;
    double bagging_ensembles_frac;
    double p_c;
    int instances_to_exchange;
    int T;
    Matrix* train_data;
    Matrix* test_data;
    Vector* train_labels;
    Vector* test_labels;
};



class BaggedBoostingEnsemble {

private:
    std::vector<std::shared_ptr<GentleBoostEnsemble>> _ensembles;
    options _opts;
    Matrix _validation_data;
    Vector _validation_labels;
    void collaborate ( );
    void collaborate_exp ( );



public:
    BaggedBoostingEnsemble ( );
    BaggedBoostingEnsemble ( const options& );
    BaggedBoostingEnsemble ( const BaggedBoostingEnsemble& );
    ~BaggedBoostingEnsemble ( );

    // setters
    void set_validation_data ( const Matrix& v ) { _validation_data = v; }
    void set_validation_labels ( const Vector& y ) { _validation_labels = y; }
    void set_opts ( const options& o ) { _opts = o; }
    void distribute_training_data ( const Matrix& );

    // getters
    const options& get_opts ( ) const { return _opts; }
    const std::vector<std::shared_ptr<GentleBoostEnsemble>>& get_ensembles( ) const { return _ensembles; }
    const Matrix& get_validation_data ( ) const { return _validation_data; }
    const Vector& get_validation_labels ( ) const { return _validation_labels; }

    // training functionalities
    void train ( const bool = true, const int = 1, const bool = true );

    // validation and prediction functionalities
    friend double diversity ( const std::vector<double>&, std::vector<double>&, void* );
    void validate ( const int );
    Vector predict_real ( const Matrix& ) const;
    Vector predict_discrete ( const Matrix& ) const;
    double compute_error_rate_real ( const Matrix&, const Vector& ) const;
    double compute_error_rate_discrete ( const Matrix&, const Vector& ) const;
    double compute_stability ( int );
    double compute_mean_emp_loss ( );
    Vector compute_margins( const Matrix&, const Vector& ) const;

    // debug
    void report_errors_discrete ( );
    void report_errors_real ( );

};


#endif // BAGGED_BOOSTING_ENSEMBLE
