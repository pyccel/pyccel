/* -------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file  */
/* or go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details. */
/* -------------------------------------------------------------------------------------- */

#ifndef         PYC_MATH_C_H
#define         PYC_MATH_C_H
#include <cassert>
#include <cmath>
#include <cstdint>
#include <complex>

/*
** (N % M) + M and fmod(N, M) + M are used to handle the negative
** operands of modulo operator.
*/

int64_t             pyc_gcd (int64_t a, int64_t b);
int64_t             pyc_lcm (int64_t a, int64_t b);

inline double       pyc_radians(double degrees)
{
    return degrees * (M_PI / 180);
}
inline double       pyc_degrees(double radians)
{
    return radians * (180.0 / M_PI);
}

template<typename Integer,
             std::enable_if_t<std::is_integral<Integer>::value, bool> = true>
Integer pyc_modulo(Integer a, Integer b)
{
    Integer modulo = a % b;
    if(!((a < 0) ^ (b < 0)) || modulo == 0)
        return modulo;
    else
        return modulo + b;
}

template<typename Floating,
             std::enable_if_t<std::is_floating_point<Floating>::value, bool> = true>
Floating pyc_modulo(Floating a, Floating b)
{
    // cppcheck-suppress invalidFunctionArg
    assert(b != 0);
    Floating modulo = std::fmod(a, b);
    if(!((a < 0) ^ (b < 0)) || modulo == 0)
        return modulo;
    else
        return modulo + b;
}

template<class T>
T sign(T x)
{
    return (x > 0) - (x < 0);
}

template<class T>
std::complex<T> sign(std::complex<T> x)
{
    T absolute = std::abs(x);
    return ((absolute == 0) ? 0.0 : (x / absolute));
}

double fpyc_bankers_round(double arg, int64_t ndigits);
int64_t ipyc_bankers_round(int64_t arg, int64_t ndigits);

template<class T>
T py_floor_div(T x, T y) {
    return x / y - ((x % y != 0) && ((x < 0) ^ (y < 0))); \
}

#endif
