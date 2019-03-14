#ifndef MULTIPRECISION_TYPES_
#define MULTIPRECISION_TYPES_

#include <Eigen/Core>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include "cpp_dec_float_50_support_eigen.hpp"

// Typedefs for multiprecision eigen matrices -- column-major by default
typedef Eigen::Matrix<boost::multiprecision::cpp_dec_float_50, Eigen::Dynamic, Eigen::Dynamic> MatrixMp;
typedef Eigen::Matrix<boost::multiprecision::cpp_dec_float_50, Eigen::Dynamic, 1> ColumnVectorMp;
typedef Eigen::Matrix<boost::multiprecision::cpp_dec_float_50, 1, Eigen::Dynamic> RowVectorMp;

// Typedefs for double precision eigen matrices -- column-major by default
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixDp;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> ColumnVectorDp;
typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowVectorDp;

namespace mp = boost::multiprecision;


#endif //MULTIPRECISION_TYPES_
