/* -------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file  */
/* or go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details. */
/* -------------------------------------------------------------------------------------- */

#ifndef         PYC_MATH_C_H
#define         PYC_MATH_C_H
#include <math.h>
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>

/*
** (N % M) + M and fmod(N, M) + M are used to handle the negative
** operands of modulo operator.
*/

int64_t             pyc_factorial(int64_t n);
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
inline int64_t      pyc_modulo(int64_t a, int64_t b)
{
        int64_t modulo = a % b;
        if(!((a < 0) ^ (b < 0)) || modulo == 0)
            return modulo;
        else
            return modulo + b;
}
inline double        pyc_fmodulo(double a, double b)
{
        double modulo = fmod(a, b);
        if(!((a < 0) ^ (b < 0)) || modulo == 0)
            return modulo;
        else
            return modulo + b;
}

long long int isign(long long int x);
double fsign(double x);
double complex csgn(double complex x);
double complex csign(double complex x);

double fpyc_bankers_round(double arg, int64_t ndigits);
int64_t ipyc_bankers_round(int64_t arg, int64_t ndigits);

double complex cpyc_expm1(double complex x);

#define PY_FLOOR_DIV_TYPE(TYPE)                         \
    static inline TYPE py_floor_div_##TYPE(TYPE x, TYPE y) { \
        return (TYPE)(x / y - ((x % y != 0) && ((x < 0) ^ (y < 0)))); \
    }

PY_FLOOR_DIV_TYPE(int8_t)
PY_FLOOR_DIV_TYPE(int16_t)
PY_FLOOR_DIV_TYPE(int32_t)
PY_FLOOR_DIV_TYPE(int64_t)

#define PY_CSIGN_TYPE(TYPE, CABS_FUNC, NAME)                         \
    static inline TYPE py_sign_type_##NAME(TYPE x) { \
    __typeof__(CABS_FUNC(x)) absolute = CABS_FUNC(x); \
    return (TYPE)((absolute == 0.0) ? (TYPE)(0.0 + 0.0 * I) : (x / (TYPE) absolute)); \
}

PY_CSIGN_TYPE(float complex, cabsf, float_complex);
PY_CSIGN_TYPE(double complex, cabs, double_complex);
PY_CSIGN_TYPE(long double complex, cabsl, long_double_complex);

inline double complex complex_min(double complex a, double complex b) {
    bool lt = creal(a) == creal(b) ? cimag(a) < cimag(b) : creal(a) < creal(b);
    return lt ? a : b;
}

inline double complex complex_max(double complex a, double complex b) {
    bool lt = creal(a) == creal(b) ? cimag(a) < cimag(b) : creal(a) < creal(b);
    return lt ? b : a;
}

#endif
