#include "bagged_boosting_ensemble.hpp"

#include <limits>
#include <algorithm>
#include <iostream>
#include <functional>
#include <thread>
#include <cstdlib>
#include <cmath>



#define MAX_MP std::numeric_limits<double>::max()


namespace bu = BaggingUtils;

/*
    Helper functions
*/
void remove_row ( Matrix& matrix, const int row_to_remove ) {
    matrix.shed_row(row_to_remove);
}

void remove_coeff ( Vector& vec, const int coeff_to_remove ) {
    vec.shed_row(coeff_to_remove);
}

void append_row ( Matrix& matrix, const Matrix& row_to_append ) {
    matrix.resize(matrix.n_rows + 1, matrix.n_cols);
    matrix.row(matrix.n_rows - 1) = row_to_append;
}

void append_coeff ( Vector& vec, double coeff_to_append ){
    vec.resize(vec.n_elem + 1);
    vec(vec.n_elem - 1) = coeff_to_append;
}

bool is_equal ( Matrix& a, Matrix& b ) {
    if(a.n_rows != b.n_rows || a.n_cols != b.n_cols)
        return false;

    for(arma::uword i = 0; i < a.n_rows; ++i) {
        for(arma::uword j = 0; j < a.n_cols; ++j) {
            if(a(i, j) != b(i, j)) return false;
        }
    }

    return true;
}



/*
    Destructor
*/
BaggedBoostingEnsemble::~BaggedBoostingEnsemble ( ) {
    for(auto&& i : _ensembles) {
        i.reset();
    }
}
/*
    Default constructor
*/
BaggedBoostingEnsemble::BaggedBoostingEnsemble ( ) { }

/*
    Constructor with options. Sets the options and subsequently parses them.
*/
BaggedBoostingEnsemble::BaggedBoostingEnsemble ( const options& opts ) {
    _opts = opts;
}

/*
    Copy constructor
*/
BaggedBoostingEnsemble::BaggedBoostingEnsemble ( const BaggedBoostingEnsemble&
    other )
{
    _opts = other._opts;
    for(decltype(other._ensembles.size())i=0; i < other._ensembles.size(); ++i){
        _ensembles.push_back(std::make_shared<GentleBoostEnsemble>(*other._ensembles[i]));
    }
}

/*
    Splits the data into training and validation sets and then:
    1) Uses Max-Flow bagging to distribute data among the ensembles
    2) Preservs the original class balance factor in each subset
*/
void BaggedBoostingEnsemble::distribute_training_data ( const Matrix& data ) {
    const std::size_t NEG = 0;
    const std::size_t POS = 1;

    bu::sampled_data s = bu::split_data(data, _opts.validation_frac);

    // create vectors of positive and negative indices for sampling
    std::vector<int> p, n;
    for(unsigned int i = 0; i < s.x_train[NEG].n_rows; ++i) n.push_back((int)i);
    for(unsigned int i = 0; i < s.x_train[POS].n_rows; ++i) p.push_back((int)i);
    int pos_sample_size = _opts.bagging_data_frac * s.x_train[POS].n_rows,
        neg_sample_size = _opts.bagging_data_frac * s.x_train[NEG].n_rows;

    // allocate data and ensembles
    if(_opts.bagging_data_frac < 1.0)
        for(int e = 0; e < _opts.n_ensembles; ++e) {
            // draw a random sample without replacement
            std::vector<int> pos_indices = bu::sampling_without_replacement(pos_sample_size, p);
            std::vector<int> neg_indices = bu::sampling_without_replacement(neg_sample_size, n);
            Matrix e_x(pos_sample_size + neg_sample_size, s.x_train[POS].n_cols);
            Vector e_y(pos_sample_size + neg_sample_size);

            unsigned int k = 0;
            for(auto idx : pos_indices) {
                e_x.row(k) = s.x_train[POS].row(idx);
                e_y(k) = s.y_train[POS](idx);
                k++;
            }

            for(auto idx : neg_indices) {
                e_x.row(k) = s.x_train[NEG].row(idx);
                e_y(k) = s.y_train[NEG](idx);
                k++;
            }
            // create the gentle boost ensemble and its base learners
            _ensembles.push_back(std::make_shared<GentleBoostEnsemble>(e_x, e_y));
            _ensembles[e]->create_base_learners();
        }
    else {
        Matrix x = arma::join_cols(s.x_train[POS], s.x_train[NEG]);
        Vector y(x.n_rows);
        int k = 0;
        for(int i = 0; i < s.y_train[POS].n_elem; i++) {
            y(k) = s.y_train[POS](i);
            k++;
        }
        for(int i = 0; i < s.y_train[NEG].n_elem; i++) {
            y(k) = s.y_train[NEG](i);
            k++;
        }
        for(int e = 0; e < _opts.n_ensembles; ++e) {
            Matrix e_x(x.n_rows, x.n_cols);
            Vector e_y(y.n_elem);
            for(int i = 0 ; i < e_x.n_rows; i++) {
                Vector rnd = arma::randi<Vector>(1, arma::distr_param(0, e_x.n_rows-1));
                e_x.row(i) = x.row(rnd(0));
                e_y(i) = y(rnd(0));
            }
            _ensembles.push_back(std::make_shared<GentleBoostEnsemble>(e_x, e_y));
            _ensembles[e]->create_base_learners();
        }
    }

    // set the validation data
    _validation_data = Matrix(s.x_valid[NEG].n_rows + s.x_valid[POS].n_rows, s.x_valid[NEG].n_cols);
    _validation_labels = Vector(s.y_valid[NEG].n_elem + s.y_valid[POS].n_elem);
    _validation_data = arma::join_cols<Matrix>(s.x_valid[NEG], s.x_valid[POS]);
    _validation_labels = arma::join_cols<Matrix>(s.y_valid[NEG], s.y_valid[POS]);

}

void step_one(unsigned int &step, arma::mat &cost, const unsigned int &N)
{
    for (unsigned int r = 0; r < N; ++r) {
        cost.row(r) -= arma::min(cost.row(r));
    }
    step = 2;
}

void step_two (unsigned int &step, const arma::mat &cost,
        arma::umat &indM, arma::ivec &rcov,
        arma::ivec &ccov, const unsigned int &N)
{
    for (unsigned int r = 0; r < N; ++r) {
        for (unsigned int c = 0; c < N; ++c) {
            if (cost.at(r, c) == 0.0 && rcov.at(r) == 0 && ccov.at(c) == 0) {
                indM.at(r, c)  = 1;
                rcov.at(r)     = 1;
                ccov.at(c)     = 1;
                break;                                              // Only take the first
                                                                    // zero in a row and column
            }
        }
    }
    /* for later reuse */
    rcov.fill(0);
    ccov.fill(0);
    step = 3;
}

void step_three(unsigned int &step, const arma::umat &indM,
        arma::ivec &ccov, const unsigned int &N)
{
    unsigned int colcount = 0;
    for (unsigned int r = 0; r < N; ++r) {
        for (unsigned int c = 0; c < N; ++c) {
            if (indM.at(r, c) == 1) {
                ccov.at(c) = 1;
            }
        }
    }
    for (unsigned int c = 0; c < N; ++c) {
        if (ccov.at(c) == 1) {
            ++colcount;
        }
    }
    if (colcount == N) {
        step = 7;
    } else {
        step = 4;
    }
}

void find_noncovered_zero(int &row, int &col,
        const arma::mat &cost, const arma::ivec &rcov,
        const arma::ivec &ccov, const unsigned int &N)
{
    unsigned int r = 0;
    unsigned int c;
    bool done = false;
    row = -1;
    col = -1;
    while (!done) {
        c = 0;
        while (true) {
            if (cost.at(r, c) == 0.0 && rcov.at(r) == 0 && ccov.at(c) == 0) {
                row = r;
                col = c;
                done = true;
            }
            ++c;
            if (c == N || done) {
                break;
            }
        }
        ++r;
        if (r == N) {
            done = true;
        }
    }
}

bool star_in_row(int &row, const arma::umat &indM,
        const unsigned int &N)
{
    bool tmp = false;
    for (unsigned int c = 0; c < N; ++c) {
        if (indM.at(row, c) == 1) {
            tmp = true;
            break;
        }
    }
    return tmp;
}

void find_star_in_row (const int &row, int &col,
        const arma::umat &indM, const unsigned int &N)
{
    col = -1;
    for (unsigned int c = 0; c < N; ++c) {
        if (indM.at(row, c) == 1) {
            col = c;
        }
    }
}

void step_four (unsigned int &step, const arma::mat &cost,
        arma::umat &indM, arma::ivec &rcov, arma::ivec &ccov,
        int &rpath_0, int &cpath_0, const unsigned int &N)
{
    int row = -1;
    int col = -1;
    bool done = false;
    while(!done) {
        find_noncovered_zero(row, col, cost, rcov,
                ccov, N);

        if (row == -1) {
            done = true;
            step = 6;
        } else {
            /* uncovered zero */
            indM(row, col) = 2;
            if (star_in_row(row, indM, N)) {
                find_star_in_row(row, col, indM, N);
                /* Cover the row with the starred zero
                 * and uncover the column with the starred
                 * zero.
                 */
                rcov.at(row) = 1;
                ccov.at(col) = 0;
            } else {
                /* No starred zero in row with
                 * uncovered zero
                 */
                done = true;
                step = 5;
                rpath_0 = row;
                cpath_0 = col;
            }
        }
    }
}

void find_star_in_col (const int &col, int &row,
        const arma::umat &indM, const unsigned int &N)
{
    row = -1;
    for (unsigned int r = 0; r < N; ++r) {
        if (indM.at(r, col) == 1) {
            row = r;
        }
    }
}

void find_prime_in_row (const int &row, int &col,
        const arma::umat &indM, const unsigned int &N)
{
    for (unsigned int c = 0; c < N; ++c) {
        if (indM.at(row, c) == 2) {
            col = c;
        }
    }
}

void augment_path (const int &path_count, arma::umat &indM,
        const arma::imat &path)
{
    for (unsigned int p = 0; p < path_count; ++p) {
        if (indM.at(path(p, 0), path(p, 1)) == 1) {
            indM.at(path(p, 0), path(p, 1)) = 0;
        } else {
            indM.at(path(p, 0), path(p, 1)) = 1;
        }
    }
}

void clear_covers (arma::ivec &rcov, arma::ivec &ccov)
{
    rcov.fill(0);
    ccov.fill(0);
}

void erase_primes(arma::umat &indM, const unsigned int &N)
{
    for (unsigned int r = 0; r < N; ++r) {
        for (unsigned int c = 0; c < N; ++c) {
            if (indM.at(r, c) == 2) {
                indM.at(r, c) = 0;
            }
        }
    }
}

void step_five (unsigned int &step,
        arma::umat &indM, arma::ivec &rcov,
        arma::ivec &ccov, arma::imat &path,
        int &rpath_0, int &cpath_0,
        const unsigned int &N)
{
    bool done = false;
    int row = -1;
    int col = -1;
    unsigned int path_count = 1;
    path.at(path_count - 1, 0) = rpath_0;
    path.at(path_count - 1, 1) = cpath_0;
    while (!done) {
        find_star_in_col(path.at(path_count - 1, 1), row,
                indM, N);
        if (row > -1) {
            /* Starred zero in row 'row' */
            ++path_count;
            path.at(path_count - 1, 0) = row;
            path.at(path_count - 1, 1) = path.at(path_count - 2, 1);
        } else {
            done = true;
        }
        if (!done) {
            /* If there is a starred zero find a primed
             * zero in this row; write index to 'col' */
            find_prime_in_row(path.at(path_count - 1, 0), col,
                    indM, N);
            ++path_count;
            path.at(path_count - 1, 0) = path.at(path_count - 2, 0);
            path.at(path_count - 1, 1) = col;
        }
    }
    augment_path(path_count, indM, path);
    clear_covers(rcov, ccov);
    erase_primes(indM, N);
    step = 3;
}

void find_smallest (double &minval, const arma::mat &cost,
        const arma::ivec &rcov, const arma::ivec &ccov,
        const unsigned int &N)
{
    for (unsigned int r = 0; r < N; ++r) {
        for (unsigned int c = 0; c < N; ++c) {
            if (rcov.at(r) == 0 && ccov.at(c) == 0) {
                if (minval > cost.at(r, c)) {
                    minval = cost.at(r, c);
                }
            }
        }
    }
}

void step_six (unsigned int &step, arma::mat &cost,
        const arma::ivec &rcov, const arma::ivec &ccov,
        const unsigned int &N)
{
    double minval = std::numeric_limits<double>::max();
    find_smallest(minval, cost, rcov, ccov, N);
    for (unsigned int r = 0; r < N; ++r) {
        for (unsigned int c = 0; c < N; ++c) {
            if (rcov.at(r) == 1) {
                cost.at(r, c) += minval;
            }
            if (ccov.at(c) == 0) {
                cost.at(r, c) -= minval;
            }
        }
    }
    step = 4;
}

arma::umat hungarian(const arma::mat &input_cost)
{
    const unsigned int N = input_cost.n_rows;
    unsigned int step = 1;
    int cpath_0 = 0;
    int rpath_0 = 0;
    arma::mat cost(input_cost);
    arma::umat indM(N, N);
    arma::ivec rcov(N);
    arma::ivec ccov(N);
    arma::imat path(2 * N, 2);

    indM = arma::zeros<arma::umat>(N, N);
    bool done = false;
    while (!done) {
        switch (step) {
            case 1:
                step_one(step, cost, N);
                break;
            case 2:
                step_two(step, cost, indM, rcov, ccov, N);
                break;
            case 3:
                step_three(step, indM, ccov, N);
                break;
            case 4:
                step_four(step, cost, indM, rcov, ccov,
                        rpath_0, cpath_0, N);
                break;
            case 5:
                step_five(step, indM, rcov, ccov,
                        path, rpath_0, cpath_0, N);
                break;
            case 6:
                step_six(step, cost, rcov, ccov, N);
                break;
            case 7:
                done = true;
                break;
        }
    }
    return indM;
}

void BaggedBoostingEnsemble::collaborate() {
    int iter = _ensembles[0]->get_iteration();

    // adaptive n_exch, i.e. exchange only while all weak learners have
    // at least one positive margin
    for(int exch = 0; exch < _opts.instances_to_exchange; exch++) {

        Matrix LORx(_opts.n_ensembles, _ensembles[0]->get_x().n_cols);
        Vector LORy(_opts.n_ensembles);
        Vector from(_opts.n_ensembles);
        Vector weight(_opts.n_ensembles);
        UVector positions(_opts.n_ensembles);
        Vector max_margins_after_removal(_opts.n_ensembles);
        unsigned int k = 0;
        for(decltype(_ensembles.size())e = 0; e < _ensembles.size(); ++e) {
            arma::uword max_pos;
            Vector margins = _ensembles[e]->get_base_learners()[iter]->compute_margins();



            // make n_exch adaptive
            UVector positive_margins_cnt = (margins >= 0.0);
            if(accu(positive_margins_cnt) == 0) return;

            Vector weighted_margins = _ensembles[e]->get_x_weights() %
                _ensembles[e]->get_base_learners()[iter]->compute_margins();

            double max_w_margin = weighted_margins.max(max_pos);
            weighted_margins(max_pos) = -std::numeric_limits<double>::infinity();
            max_margins_after_removal(e) = weighted_margins.max();

            positions(e) = max_pos;
            LORx.row(k) = _ensembles[e]->get_x().row(max_pos);
            LORy(k) = _ensembles[e]->get_y()(max_pos);
            weight(e) = _ensembles[e]->get_x_weights()(max_pos);
            from(e) = e;
            k++;
        }

        // calculate a matrix cost for minimization
        Matrix assignment_cost = arma::zeros<Matrix>(_opts.n_ensembles,
                                                     _opts.n_ensembles);

        for(unsigned int i = 0; i < LORx.n_rows; ++i)
            for(unsigned int j = 0; j < _opts.n_ensembles; ++j){
                int weighted_margin = weight(j) * _ensembles[j]->
                get_base_learners()[iter]->compute_margin(LORx.row(i), LORy(i));
                assignment_cost(i, j) =  weighted_margin-max_margins_after_removal(j);
            }

        UMatrix opt_assignment = hungarian(assignment_cost);

        for(unsigned int lor = 0; lor < opt_assignment.n_rows; ++lor) {
            int e = 0;
            while(opt_assignment(lor, e) == 0) e++;
            _ensembles[e]->get_x_nonconst().row(positions(e)) = LORx.row(lor);
            _ensembles[e]->get_y_nonconst()(positions(e)) = LORy(lor);
        }

        // retrain the regression stump
        // for(decltype(_ensembles.size())e = 0; e < _ensembles.size(); ++e)
        //     _ensembles[e]->train_single();
    }
}

double BaggedBoostingEnsemble::compute_stability( int e ) {
    // we only compute stability over positive margins !!!!!!
    // std::cout << "Iteration: " << _ensembles[e]->get_iteration() << " ";

    // first, find an instance that is not in the set (take the first from validation)
    RowVector probe_x = _validation_data.row(0);
    double probe_y = _validation_labels(0);

    // we use the current iteration
    Vector loss_before = arma::exp(-1.0 * _ensembles[e]->get_base_learners()[_ensembles[e]->get_iteration()]->compute_margins());
    Vector loss_after(_ensembles[e]->get_x().n_rows);


    for(arma::uword i = 0; i < _ensembles[e]->get_x().n_rows; ++i) {
        // if(loss_before(i) >= 1.0)  {
        //     loss_before(i) = 0.0;
        //     loss_after(i) = 0.0;
        //     continue;
        // }
        // save prior to replacement
        RowVector old_x = _ensembles[e]->get_x().row(i);
        double old_y = _ensembles[e]->get_y()(i);

        // replace
        _ensembles[e]->get_x_nonconst().row(i) = probe_x;
        _ensembles[e]->get_y_nonconst()(i) = probe_y;

        // train
        _ensembles[e]->get_base_learners()[_ensembles[e]->get_iteration()]->train();

        // get the loss on old_x
        loss_after(i) = std::exp(_ensembles[e]->get_base_learners()[_ensembles[e]->get_iteration()]->compute_margin(old_x, old_y));

        // revert x back to old x
        _ensembles[e]->get_x_nonconst().row(i) = old_x;
        _ensembles[e]->get_y_nonconst()(i) = old_y;


    }

    // re-train again to return the weak learner's state to its initial
    _ensembles[e]->get_base_learners()[_ensembles[e]->get_iteration()]->train();

    double stability = arma::mean(arma::abs(loss_before - loss_after));
    return stability;
}

double BaggedBoostingEnsemble::compute_mean_emp_loss ( ) {

    double s = 0.0;
    for(decltype(_ensembles.size())e = 0; e < _ensembles.size(); ++e) {
        s += arma::mean(arma::exp(-1.0 * _ensembles[e]->compute_margins()));
    }
    return s / (double)_opts.n_ensembles;
}

void BaggedBoostingEnsemble::collaborate_exp() {
    int iter = _ensembles[0]->get_iteration(), min2 = _opts.instances_to_exchange;
    for(decltype(_ensembles.size())e = 0; e < _ensembles.size(); ++e) {
        Vector margins_of_e = _ensembles[e]->get_base_learners()[iter]->compute_margins();
        UVector margins_of_e_positive = (margins_of_e > 0.0);
        if(accu(margins_of_e_positive) < min2)
            min2 = accu(margins_of_e_positive);
    }




    int correct_hits = 0;

    bool incorrect_prev = false;
    for(int brojac = 0; brojac < min2; brojac++) {
        int min = 1, k = 0;
        // remove "min" instances from each ensemble
        Matrix LORx(_opts.n_ensembles * min, _ensembles[0]->get_x().n_cols);
        Vector LORy(_opts.n_ensembles * min);
        Matrix weights(_opts.n_ensembles, min);

        Matrix cost(LORx.n_rows, _ensembles.size());

        //double stability_before = compute_stability(0);

        for(decltype(_ensembles.size())e = 0; e < _ensembles.size(); ++e) {
            arma::uword min_pos;
            Vector margins_of_e = _ensembles[e]->get_base_learners()[iter]->compute_margins();



            // filter out negative margins
            margins_of_e.elem(arma::find(margins_of_e <= 0.0)) += std::numeric_limits<double>::infinity();
            if(incorrect_prev) {
                margins_of_e.min(min_pos);
                margins_of_e(min_pos) = std::numeric_limits<double>::infinity();
                incorrect_prev = false;
            }
            std::vector<arma::uword> positions_of_removed_of_e;
            // find minimal margins_of_e
            for(int i = 0; i < min; i++) {
                double min_margin = margins_of_e.min(min_pos);

                margins_of_e(min_pos) = std::numeric_limits<double>::infinity();
                positions_of_removed_of_e.push_back(min_pos);

                LORx.row(k) = _ensembles[e]->get_x().row(min_pos);
                LORy(k) = _ensembles[e]->get_y()(min_pos);
                weights(e, i) = _ensembles[e]->get_x_weights()(min_pos);
                k++;


            }

            // remove instances with minimal margins_of_e, but first sort the positions in descending order
            std::sort(positions_of_removed_of_e.begin(), positions_of_removed_of_e.end(), std::greater<arma::uword>());
            for(auto&& pos_to_remove_from_e : positions_of_removed_of_e) {
                remove_row(_ensembles[e]->get_x_nonconst(), pos_to_remove_from_e);
                remove_coeff(_ensembles[e]->get_y_nonconst(), pos_to_remove_from_e);
                remove_coeff(_ensembles[e]->get_x_weights_nonconst(), pos_to_remove_from_e);
            }
        }

        // now, we construct the cost matrix
        for(arma::uword ens_j = 0; ens_j < _ensembles.size(); ++ens_j) {
            cost.col(ens_j) = 1.0 - _ensembles[ens_j]->get_base_learners()[iter]->compute_margins(LORx, LORy);
        }




        // std::cout << cost << std::endl;
        // save the cost matrix to a file "example.txt"
        // std::system("rm simplex_input");
        // cost.save("simplex_input", arma::raw_ascii);

        // // run the java program and read the result from the file simplex_result
        // int retval = std::system("java SimplexTest");
        // UMatrix optimal_assignment;
        // optimal_assignment.load("simplex_result", arma::raw_ascii);
        //
        //
        // Matrix optimal_assignment_real = arma::conv_to<Matrix>::from(optimal_assignment);
        // optimal_assignment_real = optimal_assignment_real % cost;
        // Matrix default_assignment = arma::eye(arma::size(optimal_assignment));
        // Matrix default_assignment_real = default_assignment % cost;
        // // std::cout << optimal_assignment_real << std::endl;
        // bool correct = true;
        // for(arma::uword j = 0; j < optimal_assignment.n_cols; ++j) {
        //     // std::cout << "ACCU = " << accu(optimal_assignment.col(j)) << std::endl;
        //     if(accu(optimal_assignment.col(j)) > 1) {
        //         correct = false;
        //         break;
        //     }
        //     else if(accu(optimal_assignment_real.col(j)) < accu(default_assignment_real.col(j))) {
        //         correct = false;
        //         break;
        //     }
        //
        // }
        // if(!correct) optimal_assignment.eye();
        // //std::system("rm simplex_result");
        double costb = accu(cost.diag());


        UMatrix optimal_assignment = hungarian(cost);
        Matrix oar = arma::conv_to<Matrix>::from(optimal_assignment);
        Matrix cafter = oar % cost;
        double costa = accu(oar % cost);
        for(int j = 0; j < optimal_assignment.n_cols; j++) {
            if(accu(cafter.col(j)) >= cost.diag()(j)) {
                incorrect_prev = true;
                optimal_assignment.eye();
                break;
            }
        }

        if(!incorrect_prev) correct_hits++;


        // assign the instances according to optimal_assignment
        for(arma::uword e = 0; e < _ensembles.size(); ++e) {
            // get removed weights from the e-th ensemble and sort them
            //Vector sorted_removed_weights_from_e = arma::sort(weights.row(e).t());

            // get instances from LOR that need to be added to the e-th ensemble
            // Matrix instances_to_add_to_e(min, LORx.n_cols);
            // Vector labels_to_add_to_e(min);
            arma::uword kk = 0;
            Vector weights_of_e = weights.row(e).t();
            for(arma::uword i = 0; i < optimal_assignment.n_rows; ++i)
                // if(optimal_assignment(i, e) == 1) {
                //     instances_to_add_to_e.row(kk) = LORx.row(i);
                //     labels_to_add_to_e(kk) = LORy(i);
                //     kk++;
                // }
                if(optimal_assignment(i, e) == 1) {
                    // std::cout << "ASSIGNED " <<i  << " " << e << std::endl;
                    append_row(_ensembles[e]->get_x_nonconst(), LORx.row(i));
                    append_coeff(_ensembles[e]->get_y_nonconst(), LORy(i));
                    append_coeff(_ensembles[e]->get_x_weights_nonconst(), weights_of_e(kk++));
                    for(int dim = 0; dim < _ensembles[e]->get_x().n_cols; ++dim) {
                        _ensembles[e]->_max.push_back(_ensembles[e]->get_x().col(dim).max());
                    }

                }

            // // sort the margins of the instances that need to be added to e
            // Vector margins_to_add_to_e = sorted_removed_weights_from_e % _ensembles[e]->get_base_learners()[iter]->compute_margins(instances_to_add_to_e, labels_to_add_to_e);
            // UVector sorted_margins_to_add_to_e_idx = arma::stable_sort_index(margins_to_add_to_e);
            //
            // // now we have pairs (sorted_removed_weights_from_e, sorted_margins_to_add_to_e_idx)
            // for(arma::uword i = 0; i < margins_to_add_to_e.n_elem; ++i) {
            //     append_row(_ensembles[e]->get_x_nonconst(), instances_to_add_to_e.row(sorted_margins_to_add_to_e_idx(i)));
            //     append_coeff(_ensembles[e]->get_y_nonconst(), labels_to_add_to_e(sorted_margins_to_add_to_e_idx(i)));
            //     append_coeff(_ensembles[e]->get_x_weights_nonconst(), sorted_removed_weights_from_e(i));
            // }
        }
        // std::cout << optimal_assignment%cost << std::endl;
    }



    // D O N E ! ! !

    // int iter = _ensembles[0]->get_iteration();
    //
    // // adaptive n_exch, i.e. exchange only while all weak learners have
    // // at least one positive margin
    // for(int exch = 0; exch < _opts.instances_to_exchange; exch++) {
    //
    //     Matrix LORx(_opts.n_ensembles, _ensembles[0]->get_x().n_cols);
    //     Vector LORy(_opts.n_ensembles);
    //     Vector from(_opts.n_ensembles);
    //     Vector weight(_opts.n_ensembles);
    //     UVector positions(_opts.n_ensembles);
    //     Vector min_margins_after_removal(_opts.n_ensembles);
    //     Vector max_margins_after_removal(_opts.n_ensembles);
    //
    //     unsigned int k = 0;
    //     Vector min_margins_before_removal(_opts.n_ensembles);
    //
    //     for(decltype(_ensembles.size())e = 0; e < _ensembles.size(); ++e) {
    //         arma::uword min_pos;
    //         Vector margins = _ensembles[e]->get_base_learners()[iter]->compute_margins();
    //
    //
    //
    //         // make n_exch adaptive
    //         UVector positive_margins_cnt = (margins >= 0.0);
    //         if(accu(positive_margins_cnt) == 0) return;
    //
    //         Vector weighted_margins = _ensembles[e]->get_x_weights() %
    //             _ensembles[e]->get_base_learners()[iter]->compute_margins();
    //         Vector unweighted_margins = _ensembles[e]->get_base_learners()[iter]->compute_margins();
    //
    //         // filter out the negative margins
    //         max_margins_after_removal(e) = unweighted_margins.max();
    //         weighted_margins.elem(find(weighted_margins < 0.0)) += std::numeric_limits<double>::infinity();
    //
    //         double min_w_margin = weighted_margins.min(min_pos);
    //         min_margins_before_removal(e) = min_w_margin;
    //         weighted_margins(min_pos) = std::numeric_limits<double>::infinity();
    //         min_margins_after_removal(e) = weighted_margins.min();
    //
    //         positions(e) = min_pos;
    //         LORx.row(k) = _ensembles[e]->get_x().row(min_pos);
    //         LORy(k) = _ensembles[e]->get_y()(min_pos);
    //         weight(e) = _ensembles[e]->get_x_weights()(min_pos);
    //         from(e) = e;
    //         k++;
    //     }
    //
    //     // calculate a matrix cost for minimization
    //     Matrix assignment_cost = arma::zeros<Matrix>(_opts.n_ensembles,
    //                                                  _opts.n_ensembles);
    //
    //     for(unsigned int i = 0; i < LORx.n_rows; ++i)
    //         for(unsigned int j = 0; j < _opts.n_ensembles; ++j){
    //             int weighted_margin =  weight(j) * _ensembles[j]->
    //             get_base_learners()[iter]->compute_margin(LORx.row(i), LORy(i));
    //             assignment_cost(i, j) = (max_margins_after_removal(j)-weighted_margin/weight(j));
    //         }
    //     UMatrix opt_assignment = hungarian(assignment_cost);
    //
    //     Matrix opt_assignment_real = arma::conv_to<Matrix>::from(opt_assignment);
    //     Matrix loss = assignment_cost % opt_assignment_real;
    //
    //     // ova e za da se garantira deka m-clb ke podobri
    //     if(accu(loss) > 0.0) {
    //         opt_assignment.eye();
    //     }
    //
    //     for(unsigned int lor = 0; lor < opt_assignment.n_rows; ++lor) {
    //         int e = 0;
    //         while(opt_assignment(lor, e) == 0) e++;
    //         _ensembles[e]->get_x_nonconst().row(positions(e)) = LORx.row(lor);
    //         _ensembles[e]->get_y_nonconst()(positions(e)) = LORy(lor);
    //     }
    //
    //     // std::cout << "AC = " << assignment_cost << std::endl;
    //
        // retrain the regression stump

        if(correct_hits > 0)
            for(decltype(_ensembles.size())e = 0; e < _ensembles.size(); ++e)
                _ensembles[e]->retrain_current(correct_hits);


        if(correct_hits > 0) std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
        // put everything back

    // }

}

/*
    Wrapper for trian
*/
void train_ensemble ( GentleBoostEnsemble& ens, int collab_on_each ) {
    ens.train(collab_on_each);
    ens.train_single();
}



/*
    Implements the core training functionality of the bagged boosting ensemble
*/
void BaggedBoostingEnsemble::train ( const bool collaboration, const int validation, const bool mt ) {

    int collab_on_each = (int) (1.0/_opts.p_c);

    bool after_clb = false;

    while(_ensembles.back()->get_iteration() < GentleBoostEnsemble::_T) {

        if(mt) {
            std::vector<std::thread> threads;

            for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
                threads.push_back(std::thread(&train_ensemble, std::ref(*_ensembles[e]), collab_on_each));
            }

            for(auto&& thr : threads) {
                if(thr.joinable()) {
                    thr.join();
                }
            }

            threads.clear();
        }
        else {
            if(!collaboration && collab_on_each==1) {
                for(int i = 0; i < collab_on_each  && _ensembles.back()->get_iteration() < GentleBoostEnsemble::_T; i++) {

                    for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
                        _ensembles[e]->train_single();
                        _ensembles[e]->set_iteration(_ensembles[e]->get_iteration() + 1);
                    }

                    double er, et;
                    // usde sq.err.min as default for debugging
                    if(validation == SQUARED_ERROR_MINIMIZATION) {
                        validate(SQUARED_ERROR_MINIMIZATION);
                        er = compute_error_rate_real(*_opts.train_data, *_opts.train_labels);
                        et = compute_error_rate_real(*_opts.test_data, *_opts.test_labels);
                    }
                    else if(validation == DIVERSITY_MAXIMIZATION) {
                        // validate(DIVERSITY_MAXIMIZATION);
                        // er = compute_error_rate_real(*_opts.train_data, *_opts.train_labels);
                        // et = compute_error_rate_real(*_opts.test_data, *_opts.test_labels);
                    }
                    else {
                        //report_errors_real();
                        er = compute_error_rate_real(*_opts.train_data, *_opts.train_labels);
                        et = compute_error_rate_real(*_opts.test_data, *_opts.test_labels);
                    }



                    for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e)
                        _ensembles[e]->set_weight(1.0);
                    std::cout << er << "," << et << "|";
                    report_errors_discrete();
                }
            }
                for(int i = 0; i < collab_on_each - 1 && _ensembles.back()->get_iteration() < GentleBoostEnsemble::_T; i++) {

                    for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {

                        _ensembles[e]->train_single_no_update();
                        _ensembles[e]->update_and_normalize_weights(); //ova za stability da se trgne
                        _ensembles[e]->set_iteration(_ensembles[e]->get_iteration() + 1);

                    }
                    // std::cout << "Before: " << compute_stability(0) << std::endl;



                    if(collaboration) {
                        if(validation == SQUARED_ERROR_MINIMIZATION) {
                            validate(SQUARED_ERROR_MINIMIZATION);
                            report_errors_real();

                        }
                        else if(validation == DIVERSITY_MAXIMIZATION) {
                            // validate(DIVERSITY_MAXIMIZATION);
                            // report_errors_real();
                        }
                        else{
                            report_errors_real();
                            double s = 0.0;
                            // for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e)
                            //     s+= arma::sum(arma::exp(-1.0 * _ensembles[e]->compute_margins()));
                            // std::cout << s;
                            // if(after_clb) {
                            //     std::cout << " Iteration after M-CLB" << std::endl;
                            //     after_clb = false;
                            // }
                            // else std::cout << std::endl;
                        }
                    }
                    else {
                        double er, et;
                        // usde sq.err.min as default for debugging
                        if(validation == SQUARED_ERROR_MINIMIZATION) {
                            validate(SQUARED_ERROR_MINIMIZATION);
                            er = compute_error_rate_real(*_opts.train_data, *_opts.train_labels);
                            et = compute_error_rate_real(*_opts.test_data, *_opts.test_labels);
                        }
                        else if(validation == DIVERSITY_MAXIMIZATION) {
                            // validate(DIVERSITY_MAXIMIZATION);
                            // er = compute_error_rate_real(*_opts.train_data, *_opts.train_labels);
                            // et = compute_error_rate_real(*_opts.test_data, *_opts.test_labels);
                        }
                        else {
                            er = compute_error_rate_real(*_opts.train_data, *_opts.train_labels);
                            et = compute_error_rate_real(*_opts.test_data, *_opts.test_labels);
                            // report_errors_real();

                        }

                        for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e)
                            _ensembles[e]->set_weight(1.0);

                        // bidejki gore se mesti na slednata iteracija, a stability po greska ke zeme od taa


                        // if((_ensembles[0]->get_iteration()) % collab_on_each == 0) {
                        //     // fix because iteration in B_GB is updated earlier
                        //     _ensembles[0]->set_iteration(_ensembles[0]->get_iteration()-1);
                        //     // double stability = compute_stability(0);
                        //     _ensembles[0]->set_iteration(_ensembles[0]->get_iteration()+1);
                        //     // std::cout << "Stability/Emp: " << stability << "," << compute_mean_emp_loss() << std::endl;
                        // }




                        std::cout << er << "," << et << "|";
                        report_errors_discrete();
                    }


                    // // update and normalize weights as a last step (we need this to compute stability correctly)
                    // for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
                    //     _ensembles[e]->set_iteration(_ensembles[e]->get_iteration()-1);
                    //     _ensembles[e]->update_and_normalize_weights();
                    //     _ensembles[e]->set_iteration(_ensembles[e]->get_iteration()+1);
                    // }


                }


        }

        if(collaboration) {
            double emp_before = -1;
            if(!mt) {
                for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
                    _ensembles[e]->train_single_no_update();
                    // _ensembles[e]->set_iteration(_ensembles[e]->get_iteration() -1);
                    // ova e poradi emp loss
                    // _ensembles[e]->set_iteration(_ensembles[e]->get_iteration() + 1);
                }

                // ova e poradi emp loss
                // emp_before = compute_mean_emp_loss();
                // for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
                //     _ensembles[e]->set_iteration(_ensembles[e]->get_iteration() -1);
                // }
            }

            // save
            // std::vector<Matrix> old_x;
            // std::vector<Vector> old_y;
            // std::vector<Vector> x_weights_for_next;
            //
            // for(int e = 0; e < _ensembles.size(); e++) {
            //     old_x.push_back(_ensembles[e]->get_x_copy());
            //     old_y.push_back(_ensembles[e]->get_y_copy());
            //     Vector w = _ensembles[e]->get_x_weights_copy();
            //     w = w % arma::exp(-1.0 * _ensembles[e]->get_base_learners()[_ensembles[e]->get_iteration()]->compute_margins());
            //     w /= arma::sum(w);
            //     x_weights_for_next.push_back(w);
            //
            // }


            // COLLABORATION
            // double stability_before = compute_stability(0);
            collaborate_exp();
            // double stability_after = compute_stability(0);

            for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
                // _ensembles[e]->set_x(old_x[e]);
                // _ensembles[e]->set_y(old_y[e]);
                // _ensembles[e]->set_x_weights(x_weights_for_next[e]);
                _ensembles[e]->update_and_normalize_weights();
                _ensembles[e]->set_iteration(_ensembles[e]->get_iteration() + 1);
                after_clb = true;
            }
            // double emp_after = compute_mean_emp_loss();
            // std::cout << "Stability: " << stability_before << "," << stability_after;
            // std::cout<< " EmpErr: " << emp_before << "," << emp_after << std::endl;

            if(validation == SQUARED_ERROR_MINIMIZATION) {
                validate(SQUARED_ERROR_MINIMIZATION);
                report_errors_real();
            }
            else if(validation == DIVERSITY_MAXIMIZATION) {
                // validate(DIVERSITY_MAXIMIZATION);
                // report_errors_real();
            }
            else {
                report_errors_real();
                // double s = 0.0;
                // for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e)
                //     s+= arma::sum(arma::exp(-1.0 * _ensembles[e]->compute_margins()));
                // std::cout << s << " M-CLB" << std::endl;
            }
        }

    }
}

// void BaggedBoostingEnsemble::train ( const bool collaboration, const int validation, const bool mt ) {
//
//     int collab_on_each = (int) (1.0/_opts.p_c);
//
//     bool after_clb = false;
//
//     while(_ensembles.back()->get_iteration() < GentleBoostEnsemble::_T) {
//
//         if(mt) {
//             std::vector<std::thread> threads;
//
//             for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
//                 threads.push_back(std::thread(&train_ensemble, std::ref(*_ensembles[e]), collab_on_each));
//             }
//
//             for(auto&& thr : threads) {
//                 if(thr.joinable()) {
//                     thr.join();
//                 }
//             }
//
//             threads.clear();
//         }
//         else {
//             if(!collaboration && collab_on_each==1) {
//                 for(int i = 0; i < collab_on_each  && _ensembles.back()->get_iteration() < GentleBoostEnsemble::_T; i++) {
//
//                     for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
//                         _ensembles[e]->train_single();
//                         //_ensembles[e]->set_iteration(_ensembles[e]->get_iteration() + 1);
//                     }
//
//                     double er, et;
//                     // usde sq.err.min as default for debugging
//                     if(validation == SQUARED_ERROR_MINIMIZATION) {
//                         validate(SQUARED_ERROR_MINIMIZATION);
//                         er = compute_error_rate_real(*_opts.train_data, *_opts.train_labels);
//                         et = compute_error_rate_real(*_opts.test_data, *_opts.test_labels);
//                     }
//                     else if(validation == DIVERSITY_MAXIMIZATION) {
//                         // validate(DIVERSITY_MAXIMIZATION);
//                         // er = compute_error_rate_real(*_opts.train_data, *_opts.train_labels);
//                         // et = compute_error_rate_real(*_opts.test_data, *_opts.test_labels);
//                     }
//                     else {
//                         //report_errors_real();
//                         er = compute_error_rate_real(*_opts.train_data, *_opts.train_labels);
//                         et = compute_error_rate_real(*_opts.test_data, *_opts.test_labels);
//                     }
//
//
//
//                     for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e)
//                         _ensembles[e]->set_weight(1.0);
//                     // std::cout << er << "," << et << "|";
//                     // report_errors_discrete();
//                 }
//             }
//             for(int i = 0; i < collab_on_each - 1 && _ensembles.back()->get_iteration() < GentleBoostEnsemble::_T; i++) {
//
//                 for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
//
//                     _ensembles[e]->train_single_no_update();
//                     // _ensembles[e]->update_and_normalize_weights(); //ova za stability da se trgne
//                     _ensembles[e]->set_iteration(_ensembles[e]->get_iteration() + 1);
//
//                 }
//
//
//
//                 if(collaboration) {
//                     if(validation == SQUARED_ERROR_MINIMIZATION) {
//                         validate(SQUARED_ERROR_MINIMIZATION);
//                         // report_errors_real();
//
//                     }
//                     else if(validation == DIVERSITY_MAXIMIZATION) {
//                         // validate(DIVERSITY_MAXIMIZATION);
//                         // report_errors_real();
//                     }
//                     else{
//                         // report_errors_real();
//                     }
//                 }
//                 else {
//                     double er, et;
//                     // usde sq.err.min as default for debugging
//                     if(validation == SQUARED_ERROR_MINIMIZATION) {
//                         validate(SQUARED_ERROR_MINIMIZATION);
//                         er = compute_error_rate_real(*_opts.train_data, *_opts.train_labels);
//                         et = compute_error_rate_real(*_opts.test_data, *_opts.test_labels);
//                     }
//                     else if(validation == DIVERSITY_MAXIMIZATION) {
//                         // validate(DIVERSITY_MAXIMIZATION);
//                         // er = compute_error_rate_real(*_opts.train_data, *_opts.train_labels);
//                         // et = compute_error_rate_real(*_opts.test_data, *_opts.test_labels);
//                     }
//                     else {
//                         er = compute_error_rate_real(*_opts.train_data, *_opts.train_labels);
//                         et = compute_error_rate_real(*_opts.test_data, *_opts.test_labels);
//                         // report_errors_real();
//
//                     }
//
//                     for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e)
//                         _ensembles[e]->set_weight(1.0);
//
//                     // bidejki gore se mesti na slednata iteracija, a stability po greska ke zeme od taa
//
//
//                     if((_ensembles[0]->get_iteration()) % collab_on_each == 0) {
//                         // fix because iteration in B_GB is updated earlier
//                         _ensembles[0]->set_iteration(_ensembles[0]->get_iteration()-1);
//                         double stability = compute_stability(0);
//                         _ensembles[0]->set_iteration(_ensembles[0]->get_iteration()+1);
//                         std::cout << "Stability/Emp: " << stability << "," << compute_mean_emp_loss() << std::endl;
//                     }
//
//
//
//
//                     // std::cout << er << "," << et << "|";
//                     // report_errors_discrete();
//                 }
//
//
//                 // update and normalize weights as a last step (we need this to compute stability correctly)
//                 for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
//                     _ensembles[e]->set_iteration(_ensembles[e]->get_iteration()-1);
//                     _ensembles[e]->update_and_normalize_weights();
//                     _ensembles[e]->set_iteration(_ensembles[e]->get_iteration()+1);
//                 }
//
//
//             }
//         }
//
//         if(collaboration) {
//             double emp_before = -1;
//             if(!mt) {
//                 for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
//                     _ensembles[e]->train_single_no_update();
//                     // _ensembles[e]->set_iteration(_ensembles[e]->get_iteration() -1);
//                     // ova e poradi emp loss
//                     _ensembles[e]->set_iteration(_ensembles[e]->get_iteration() + 1);
//                 }
//
//                 // ova e poradi emp loss
//                 emp_before = compute_mean_emp_loss();
//                 for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
//                     _ensembles[e]->set_iteration(_ensembles[e]->get_iteration() -1);
//                 }
//             }
//
//             // save
//             // std::vector<Matrix> old_x;
//             // std::vector<Vector> old_y;
//             // std::vector<Vector> x_weights_for_next;
//             //
//             // for(int e = 0; e < _ensembles.size(); e++) {
//             //     old_x.push_back(_ensembles[e]->get_x_copy());
//             //     old_y.push_back(_ensembles[e]->get_y_copy());
//             //     Vector w = _ensembles[e]->get_x_weights_copy();
//             //     w = w % arma::exp(-1.0 * _ensembles[e]->get_base_learners()[_ensembles[e]->get_iteration()]->compute_margins());
//             //     w /= arma::sum(w);
//             //     x_weights_for_next.push_back(w);
//             //
//             // }
//
//
//             // COLLABORATION
//             double stability_before = compute_stability(0);
//             collaborate_exp();
//             double stability_after = compute_stability(0);
//
//             for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
//                 // _ensembles[e]->set_x(old_x[e]);
//                 // _ensembles[e]->set_y(old_y[e]);
//                 // _ensembles[e]->set_x_weights(x_weights_for_next[e]);
//                 _ensembles[e]->update_and_normalize_weights();
//                 _ensembles[e]->set_iteration(_ensembles[e]->get_iteration() + 1);
//                 after_clb = true;
//             }
//             double emp_after = compute_mean_emp_loss();
//             std::cout << "Stability: " << stability_before << "," << stability_after;
//             std::cout<< " EmpErr: " << emp_before << "," << emp_after << std::endl;
//
//             if(validation == SQUARED_ERROR_MINIMIZATION) {
//                 validate(SQUARED_ERROR_MINIMIZATION);
//                 // report_errors_real();
//             }
//             else if(validation == DIVERSITY_MAXIMIZATION) {
//                 // validate(DIVERSITY_MAXIMIZATION);
//                 // report_errors_real();
//             }
//             else {
//                 // report_errors_real();
//                 // double s = 0.0;
//                 // for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e)
//                 //     s+= arma::sum(arma::exp(-1.0 * _ensembles[e]->compute_margins()));
//                 // std::cout << s << " M-CLB" << std::endl;
//             }
//         }
//
//     }
// }

/*
    non-member diversity function
*/

double diversity ( const std::vector<double>& w,
    std::vector<double>& grad, void* func_data )
{

    BaggedBoostingEnsemble* ens = (BaggedBoostingEnsemble*) func_data;
    Matrix margins(ens->_validation_data.n_rows, ens->_opts.n_ensembles);

    for(decltype(ens->_ensembles.size()) e = 0; e < ens->_ensembles.size(); ++e) {
            margins.col(e) = ens->_ensembles[e]->compute_margins(ens->_validation_data,
            ens->_validation_labels);
    }

    RowVector w_row(w.size());
    for(unsigned int i = 0; i < w_row.size(); ++i) w_row(i) = w[i];

    Matrix margins_sigmoid(margins.n_rows, margins.n_cols);
    for(unsigned int i = 0; i < margins.n_rows; ++i) {
        RowVector temp = margins.row(i) % w_row;
        margins_sigmoid.row(i) = arma::sigmoid(temp);
    }
    Vector margins_sigmoid_mean = mean(margins_sigmoid, 1);

    if(!grad.empty()) {

        for(decltype(grad.size()) i = 0; i < grad.size(); ++i) {
            Vector inner_sum = arma::zeros<Vector>(ens->_validation_data.n_rows);

            for(unsigned int j = 0; j < w.size(); ++j) {

                if(j != i) {
                    inner_sum +=
                    ((double)(-2) / (double)(w.size())) *
                    (margins_sigmoid.col(j) - margins_sigmoid_mean) % (
                    margins.col(i) % (margins_sigmoid.col(i)) % (
                    (-margins_sigmoid.col(i) + 1.0)));
                }
                else {
                    inner_sum +=
                    ((double)(2) * ((double)(w.size() - 1))/
                    (double)(w.size())) * (margins_sigmoid.col(j) -
                    margins_sigmoid_mean) % (margins.col(i) % (
                    margins_sigmoid.col(i)) % (
                    (-margins_sigmoid.col(i) + 1.0)));
                }
            }

            grad[i] = (1.0 / (double)(w.size()-1)) *
            arma::accu(inner_sum);
        }

    }

    double variance = arma::accu(stddev(margins_sigmoid, 0, 1));

    return variance;
}

/*
    Implements a validation method
*/
void BaggedBoostingEnsemble::validate ( const int METHOD )
{
    if(METHOD == SQUARED_ERROR_MINIMIZATION) {

        // first compute the outputs on the validations set
        Matrix f_valid(_validation_data.n_rows, _opts.n_ensembles);
        Matrix G(_opts.n_ensembles, _opts.n_ensembles);
        Vector b(_opts.n_ensembles);

        for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
            f_valid.col(e) = _ensembles[e]->predict_real(_validation_data);
        }

        for(unsigned int j = 0; j < _opts.n_ensembles; ++j) {
            for(unsigned int k = 0; k < _opts.n_ensembles; ++k) {
                G(j, k) = dot(f_valid.col(j), f_valid.col(k));
            }
            b(j) = dot(_validation_labels, f_valid.col(j));
        }

        Vector ensemble_weights = arma::solve(G, b);
        for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
                _ensembles[e]->set_weight(ensemble_weights(e));
        }
        return;
    }


}

/*
    Predicts the output labels using real outputs from the gentle boost ensembles
*/
Vector BaggedBoostingEnsemble::predict_real ( const Matrix& x ) const {
    Vector y = arma::zeros<Vector>(x.n_rows);
    for(auto&& ens_ptr : _ensembles) {
        y += ens_ptr->get_weight() * ens_ptr->predict_real(x);
    }
    return arma::sign(y);
}

Vector BaggedBoostingEnsemble::compute_margins( const Matrix& x, const Vector& y ) const {
    Vector o = arma::zeros<Vector>(x.n_rows);
    for(auto&& ens_ptr : _ensembles) {
        o += ens_ptr->get_weight() * ens_ptr->predict_real(x);
    }
    return y % o;
}

/*
    Predicts the output labels using discrete outputs from the gentle boost ensembles
*/
Vector BaggedBoostingEnsemble::predict_discrete ( const Matrix& x ) const {
    Vector y = arma::zeros<Vector>(x.n_rows);
    for(auto&& ens_ptr : _ensembles) {
        y += ens_ptr->get_weight() * ens_ptr->predict_discrete(x);
    }
    return sign(y);
}

/*
    Computes the error (misclassification rate) in percent
*/
double BaggedBoostingEnsemble::compute_error_rate_real ( const Matrix& x, const Vector& y ) const {
    Vector f = predict_real(x);
    int misclassified = sum(f != y);

    return ((double) misclassified / (double) y.n_elem);

}

/*
    Computes the error (misclassification rate) in percent
*/
double BaggedBoostingEnsemble::compute_error_rate_discrete ( const Matrix& x, const Vector& y ) const {
    Vector f = predict_discrete(x);
    int misclassified = sum(f != y);

    return ((double) misclassified / (double) y.n_elem);
}

/*
    Reports errors to the standard output
*/
void BaggedBoostingEnsemble::report_errors_discrete ( ) {

    if(_opts.train_data != nullptr && _opts.train_labels != nullptr) {
        double train_err = compute_error_rate_discrete(*_opts.train_data, *_opts.train_labels);
        std::cout << train_err << ",";
    }
    if(_opts.test_data != nullptr && _opts.test_labels != nullptr) {
        double test_err = compute_error_rate_discrete(*_opts.test_data, *_opts.test_labels);
        std::cout << test_err << std::endl;
    }
}

void BaggedBoostingEnsemble::report_errors_real ( ) {

    if(_opts.train_data != nullptr && _opts.train_labels != nullptr) {
        double train_err = compute_error_rate_real(*_opts.train_data, *_opts.train_labels);
        std::cout << train_err << ",";
    }
    if(_opts.test_data != nullptr && _opts.test_labels != nullptr) {
        double test_err = compute_error_rate_real(*_opts.test_data, *_opts.test_labels);
        std::cout << test_err << std::endl;
    }
}









/*
Implements the inter-ensemble collaboration procedure
*/
// void BaggedBoostingEnsemble::collaborate ( ) {
//     std::vector<std::vector<unsigned int>> max_margin_indices(
//         _opts.instances_to_exchange);
//     Matrix max_margins(_opts.instances_to_exchange, _opts.n_ensembles);
//
//     for(auto&& i : max_margin_indices) {
//         i = std::vector<unsigned int>(_opts.n_ensembles);
//     }
//
//     for(decltype(_ensembles.size()) j = 0; j < _ensembles.size(); ++j) {
//
//         Vector weights_for_margins = _ensembles[j]->get_x_weights();
//
//         Vector margins = _ensembles[j]->get_base_learners()
//             [_ensembles[j]->get_iteration()]->compute_margins();
//
//         Vector weighted_margins = weights_for_margins % (margins);
//
//         int to_exch = 0;
//         while(to_exch < _opts.instances_to_exchange) {
//             max_margins(to_exch, j) = weighted_margins.max(max_margin_indices[to_exch][j]);
//             weighted_margins(max_margin_indices[to_exch][j]) =
//                 (double)(std::numeric_limits<double>::min());
//             to_exch++;
//         }
//     }
//
//     // list of removed instances
//     Matrix LOR(_opts.n_ensembles * _opts.instances_to_exchange,
//         _ensembles[0]->get_x().n_cols);
//     Vector LOR_y(_opts.n_ensembles * _opts.instances_to_exchange);
//     std::vector<int> index_of;
//     std::vector<double> weight_of;
//     std::vector<int> subset_of;
//     int row_ind = 0;
//     for(decltype(max_margin_indices[0].size()) j = 0; j < max_margin_indices[0].
//         size(); ++j) {
//         for(decltype(max_margin_indices.size()) i = 0; i < max_margin_indices.
//             size(); ++i) {
//
//             LOR.row(row_ind) = _ensembles[j]->get_x().row(
//                 max_margin_indices[i][j]);
//             LOR_y(row_ind) = _ensembles[j]->get_y()(max_margin_indices[i][j]);
//             index_of.push_back(max_margin_indices[i][j]);
//             subset_of.push_back(j);
//             weight_of.push_back(_ensembles[j]->get_x_weights()(max_margin_indices[i][j]));
//             row_ind++;
//         }
//     }
//
//     // remove the rows from each subset
//     Matrix removed_weights(_opts.instances_to_exchange, _opts.n_ensembles);
//
//     for(decltype(max_margin_indices[0].size()) j = 0; j < max_margin_indices[0].
//         size(); ++j) {
//         std::vector<int> j_th_rows;
//         for(decltype(max_margin_indices.size()) i = 0; i < max_margin_indices.
//             size(); ++i) {
//             j_th_rows.push_back(max_margin_indices[i][j]);
//         }
//         // sort the indices in descending order
//         std::sort(j_th_rows.begin(), j_th_rows.end(), std::greater<int>());
//         // remove the rows
//
//         for(auto ind_to_remove : j_th_rows) {
//             remove_row(_ensembles[j]->get_x_nonconst(), ind_to_remove);
//             remove_coeff(_ensembles[j]->get_y_nonconst(), ind_to_remove);
//             remove_coeff(_ensembles[j]->get_x_weights_nonconst(),ind_to_remove);
//         }
//     }
//
//     // re-train the current weak learner in each ensemble
//     for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
//         // _ensembles[e]->normalize_weights();
//         // _ensembles[e]->retrain_current();
//     }
//
//
//
//     // at this point, each ensemble must receive exactly 'instances_to_exchange'
//     // new, unseen instances from any of the rest in LOR
//     std::vector<int> num_received(_ensembles.size(), 0);
//     std::vector<bool> is_taken(LOR.n_rows, false);
//     Vector new_min_margins(_ensembles.size());
//
//     for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
//         // recalculate the minimum weighted margins for each ensemble (weak)
//         Vector weights_for_margins = _ensembles[e]->get_x_weights();
//
//         Vector margins = _ensembles[e]->get_base_learners()
//             [_ensembles[e]->get_iteration()]->compute_margins();
//
//
//
//         Vector weighted_margins = weights_for_margins % (margins);
//
//         double min_margin = weighted_margins.max();
//         new_min_margins(e) = min_margin;
//
//         // now, the e-th ensemble tries to get at most 'instances_to_exchange'
//         // new instances
//         for(unsigned int i = 0; i < LOR.n_rows && num_received[e] <
//             _opts.instances_to_exchange; ++i)
//         {
//             if((unsigned)subset_of[i] != e) {
//                 // compute the margin of the instance as a test instance with
//                 // an estimated uniform weight
//                 double margin = _ensembles[e]->get_base_learners()
//                 [_ensembles[e]->get_iteration()]->compute_margin(
//                     LOR.row(i), LOR_y(i));
//
//                 // estimated normalized weight upon adding the instance
//                 double weight = weight_of[i];
//
//
//                 margin *= weight;
//
//                 if(margin <= (min_margin + (double)(
//                     std::numeric_limits<double>::epsilon()))) {
//
//                     // take the instance
//                     append_row(_ensembles[e]->get_x_nonconst(), LOR.row(i));
//                     append_coeff(_ensembles[e]->get_y_nonconst(), LOR_y(i));
//                     append_coeff(_ensembles[e]->get_x_weights_nonconst(),
//                         1.0 /
//                         (double)(_ensembles[e]->get_x().n_rows +
//                         _opts.instances_to_exchange - (num_received[e] + 1)));
//
//                     ++num_received[e];
//                     is_taken[i] = true;
//                 }
//             }
//         }
//     }
//
//
//     // at this point, all untaken instances are redistributed to a best-fit ens.
//
//     for(unsigned int i = 0; i < LOR.n_rows; ++i) {
//         if(!is_taken[i]) {
//
//             decltype(_ensembles.size()) best_fit_e = -1;
//             double best_offset = MAX_MP;
//
//             for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
//                 if(num_received[e] < _opts.instances_to_exchange) {
//
//                     double margin = _ensembles[e]->
//                     get_base_learners()[_ensembles[e]->get_iteration()]->
//                     compute_margin(LOR.row(i), LOR_y(i));
//
//                     // estimated normalized weight upon adding the instance
//                     double weight = weight_of[i];
//
//                     margin *= weight;
//
//
//                     if(margin - new_min_margins(e) < best_offset) {
//                         best_offset = new_min_margins(e) - margin;
//                         best_fit_e = e;
//                     }
//                 }
//             }
//
//             // add the instance to the best-fit ensemble (if possible)
//             if(best_fit_e != -1) {
//                 append_row(_ensembles[best_fit_e]->get_x_nonconst(), LOR.row(i));
//                 append_coeff(_ensembles[best_fit_e]->get_y_nonconst(), LOR_y(i));
//                 append_coeff(_ensembles[best_fit_e]->get_x_weights_nonconst(),removed_weights(i, best_fit_e));
//
//                 is_taken[i] = true;
//                 ++num_received[best_fit_e];
//             }
//         }
//     }
//
//     // re-train the current weak learner in each ensemble and prepare for next i
//     for(decltype(_ensembles.size()) e = 0; e < _ensembles.size(); ++e) {
//         _ensembles[e]->update_max();
//         _ensembles[e]->normalize_weights();
//         _ensembles[e]->train_single();
//         _ensembles[e]->set_iteration(_ensembles[e]->get_iteration() + 1);
//     }
// }
