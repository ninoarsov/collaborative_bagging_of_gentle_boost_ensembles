#ifndef CPP_DEC_FLOAT_50_SUPPORT_EIGEN_
#define CPP_DEC_FLOAT_50_SUPPORT_EIGEN_

#include <Eigen/Core>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>

namespace Eigen {

template<> struct NumTraits<boost::multiprecision::cpp_dec_float_50>
    : NumTraits<double>
{
    typedef boost::multiprecision::cpp_dec_float_50 Real;
    typedef boost::multiprecision::cpp_dec_float_50 NonInteger;
    typedef boost::multiprecision::cpp_dec_float_50 Nested;
    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 3,
        MulCost = 3
    };
};

}

namespace boost {
namespace multiprecision {

inline const boost::multiprecision::cpp_dec_float_50& conj(const boost::multiprecision::cpp_dec_float_50& x) { return x; }
inline const boost::multiprecision::cpp_dec_float_50& real(const boost::multiprecision::cpp_dec_float_50& x) { return x; }
inline boost::multiprecision::cpp_dec_float_50 imag(const boost::multiprecision::cpp_dec_float_50& x) { return 0.; }
inline boost::multiprecision::cpp_dec_float_50 abs2(const boost::multiprecision::cpp_dec_float_50& x) { return x*x; }
inline boost::multiprecision::cpp_dec_float_50 expo(const boost::multiprecision::cpp_dec_float_50& x) { return exp(x); }
inline boost::multiprecision::cpp_dec_float_50 sign(const boost::multiprecision::cpp_dec_float_50& x) {
    return x < boost::multiprecision::cpp_dec_float_50(0) ? boost::multiprecision::cpp_dec_float_50(-1) :boost::multiprecision::cpp_dec_float_50(1);
}
inline boost::multiprecision::cpp_dec_float_50 sigmoid(const boost::multiprecision::cpp_dec_float_50& x) {
    return boost::multiprecision::cpp_dec_float_50(1) / (cpp_dec_float_50(1) + exp(-x));
}

}
}

#endif //CPP_DEC_FLOAT_50_SUPPORT_EIGEN
