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

#include <stc/cspan.h>
#define STC_CSPAN_INDEX_TYPE int64_t
#ifndef _ARRAY_INT8_T_1D
#define _ARRAY_INT8_T_1D
using_cspan(array_int8_t_1d, int8_t, 1);
#endif // _ARRAY_INT8_T_1D
#ifndef _ARRAY_INT16_T_1D
#define _ARRAY_INT16_T_1D
using_cspan(array_int16_t_1d, int16_t, 1);
#endif // _ARRAY_INT16_T_1D
#ifndef _ARRAY_INT32_T_1D
#define _ARRAY_INT32_T_1D
using_cspan(array_int32_t_1d, int32_t, 1);
#endif // _ARRAY_INT32_T_1D
#ifndef _ARRAY_INT64_T_1D
#define _ARRAY_INT64_T_1D
using_cspan(array_int64_t_1d, int64_t, 1);
#endif // _ARRAY_INT64_T_1D
#ifndef _ARRAY_FLOAT_1D
#define _ARRAY_FLOAT_1D
using_cspan(array_float_1d, float, 1);
#endif // _ARRAY_FLOAT_1D
#ifndef _ARRAY_DOUBLE_1D
#define _ARRAY_DOUBLE_1D
using_cspan(array_double_1d, double, 1);
#endif // _ARRAY_DOUBLE_1D
#ifndef _ARRAY_FLOAT_COMPLEX_1D
#define _ARRAY_FLOAT_COMPLEX_1D
using_cspan(array_float_complex_1d, float complex, 1);
#endif // _ARRAY_FLOAT_COMPLEX_1D
#ifndef _ARRAY_DOUBLE_COMPLEX_1D
#define _ARRAY_DOUBLE_COMPLEX_1D
using_cspan(array_double_complex_1d, double complex, 1);
#endif // _ARRAY_DOUBLE_COMPLEX_1D
#ifndef _ARRAY_INT8_T_2D
#define _ARRAY_INT8_T_2D
using_cspan(array_int8_t_2d, int8_t, 2);
#endif // _ARRAY_INT8_T_2D
#ifndef _ARRAY_INT16_T_2D
#define _ARRAY_INT16_T_2D
using_cspan(array_int16_t_2d, int16_t, 2);
#endif // _ARRAY_INT16_T_2D
#ifndef _ARRAY_INT32_T_2D
#define _ARRAY_INT32_T_2D
using_cspan(array_int32_t_2d, int32_t, 2);
#endif // _ARRAY_INT32_T_2D
#ifndef _ARRAY_INT64_T_2D
#define _ARRAY_INT64_T_2D
using_cspan(array_int64_t_2d, int64_t, 2);
#endif // _ARRAY_INT64_T_2D
#ifndef _ARRAY_FLOAT_2D
#define _ARRAY_FLOAT_2D
using_cspan(array_float_2d, float, 2);
#endif // _ARRAY_FLOAT_2D
#ifndef _ARRAY_DOUBLE_2D
#define _ARRAY_DOUBLE_2D
using_cspan(array_double_2d, double, 2);
#endif // _ARRAY_DOUBLE_2D
#ifndef _ARRAY_FLOAT_COMPLEX_2D
#define _ARRAY_FLOAT_COMPLEX_2D
using_cspan(array_float_complex_2d, float complex, 2);
#endif // _ARRAY_FLOAT_COMPLEX_2D
#ifndef _ARRAY_DOUBLE_COMPLEX_2D
#define _ARRAY_DOUBLE_COMPLEX_2D
using_cspan(array_double_complex_2d, double complex, 2);
#endif // _ARRAY_DOUBLE_COMPLEX_2D


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

// Macro to generate precision-specific sign functions for integer and floating-point types
#define PY_SIGN_TYPE(TYPE, NAME) \
    static inline TYPE py_sign_type_##NAME(TYPE x) { \
        return (TYPE)((x > 0) - (x < 0)); \
    }

// Instantiate integer sign functions for all precisions
PY_SIGN_TYPE(int8_t, int8_t);
PY_SIGN_TYPE(int16_t, int16_t);
PY_SIGN_TYPE(int32_t, int32_t);
PY_SIGN_TYPE(int64_t, int64_t);

// Instantiate floating-point sign functions for all precisions
PY_SIGN_TYPE(float, float);
PY_SIGN_TYPE(double, double);
PY_SIGN_TYPE(long double, long_double);

void pyc_matmul_array_int8_t_2d(array_int8_t_2d out, array_int8_t_2d A, array_int8_t_2d x);
void pyc_matmul_array_int16_t_2d(array_int16_t_2d out, array_int16_t_2d A, array_int16_t_2d x);
void pyc_matmul_array_int32_t_2d(array_int32_t_2d out, array_int32_t_2d A, array_int32_t_2d x);
void pyc_matmul_array_int64_t_2d(array_int64_t_2d out, array_int64_t_2d A, array_int64_t_2d x);
void pyc_matmul_array_float_2d(array_float_2d out, array_float_2d A, array_float_2d x);
void pyc_matmul_array_double_2d(array_double_2d out, array_double_2d A, array_double_2d x);
void pyc_matmul_array_float_complex_2d(array_float_complex_2d out, array_float_complex_2d A, array_float_complex_2d x);
void pyc_matmul_array_double_complex_2d(array_double_complex_2d out, array_double_complex_2d A, array_double_complex_2d x);

void pyc_matvecmul_int8_t(array_int8_t_1d out, array_int8_t_2d A, array_int8_t_1d x);
void pyc_matvecmul_int16_t(array_int16_t_1d out, array_int16_t_2d A, array_int16_t_1d x);
void pyc_matvecmul_int32_t(array_int32_t_1d out, array_int32_t_2d A, array_int32_t_1d x);
void pyc_matvecmul_int64_t(array_int64_t_1d out, array_int64_t_2d A, array_int64_t_1d x);
void pyc_matvecmul_float(array_float_1d out, array_float_2d A, array_float_1d x);
void pyc_matvecmul_double(array_double_1d out, array_double_2d A, array_double_1d x);
void pyc_matvecmul_float_complex(array_float_complex_1d out, array_float_complex_2d A, array_float_complex_1d x);
void pyc_matvecmul_double_complex(array_double_complex_1d out, array_double_complex_2d A, array_double_complex_1d x);

void pyc_vecmatmul_int8_t(array_int8_t_1d out, array_int8_t_1d A, array_int8_t_2d x);
void pyc_vecmatmul_int16_t(array_int16_t_1d out, array_int16_t_1d A, array_int16_t_2d x);
void pyc_vecmatmul_int32_t(array_int32_t_1d out, array_int32_t_1d A, array_int32_t_2d x);
void pyc_vecmatmul_int64_t(array_int64_t_1d out, array_int64_t_1d A, array_int64_t_2d x);
void pyc_vecmatmul_float(array_float_1d out, array_float_1d A, array_float_2d x);
void pyc_vecmatmul_double(array_double_1d out, array_double_1d A, array_double_2d x);
void pyc_vecmatmul_float_complex(array_float_complex_1d out, array_float_complex_1d A, array_float_complex_2d x);
void pyc_vecmatmul_double_complex(array_double_complex_1d out, array_double_complex_1d A, array_double_complex_2d x);

#endif
