/* -------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file  */
/* or go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details. */
/* -------------------------------------------------------------------------------------- */

#include "pyc_math_c.h"

/*---------------------------------------------------------------------------*/
int64_t pyc_factorial(int64_t n)
{
    int64_t res = 1;

    /* ValueError: factorial() not defined for negative values */
    if (n < 0)
        return 0;
    for (int64_t i = 2; i <= n; i++)
        res *= i;
    return (res);
}
/*---------------------------------------------------------------------------*/
int64_t pyc_gcd(int64_t a, int64_t b)
{
    a = a > 0 ? a : -a;
    b = b > 0 ? b : -b;
    while (b)
    {
        a %= b;
        /* swap a and b*/
        a = a ^ b;
        b = b ^ a;
        a = a ^ b;
    }
    return a;
}
/*---------------------------------------------------------------------------*/
int64_t pyc_lcm(int64_t a, int64_t b)
{
    a = a > 0 ? a : -a;
    b = b > 0 ? b : -b;
    return a / pyc_gcd(a, b) * b;
}
/*---------------------------------------------------------------------------*/

/* numpy.sign for float, double and integers */
long long int isign(long long int x)
{
    return (x > 0) - (x < 0);
}

/* numpy.sign for float, double and integers */
double fsign(double x)
{
    return (double)((x > 0) - (x < 0));
}

/* numpy.sign for complex for NumPy v1 */
double complex csgn(double complex x)
{
    return x ? ((!creal(x) && cimag(x) < 0) || (creal(x) < 0) ? -1 : 1) : 0;
}

double complex csign(double complex x)
{
    double absolute = cabs(x);
    return ((absolute == 0) ? 0.0 : (x / absolute));
}

/*---------------------------------------------------------------------------*/

double fpyc_bankers_round(double arg, int64_t ndigits)
{
    double factor = pow(10.0, ndigits);
    arg *= factor;

    double nearest_int_fix = copysign(0.5, arg);

    double rnd = (int64_t)(arg + nearest_int_fix);

    double diff = arg - rnd;

    if (ndigits <= 0 && (diff == 0.5 || diff == -0.5))
    {
        rnd = ((int64_t)(arg * 0.5 + nearest_int_fix)) * 2.0;
    }

    return rnd / factor;
}

int64_t ipyc_bankers_round(int64_t arg, int64_t ndigits)
{
    if (ndigits >= 0)
    {
        return arg;
    }
    else
    {
        int64_t mul_fact = 1;
        for (int i = 0; i < -ndigits; ++i)
            mul_fact *= 10;

        int64_t pivot_point = copysign(5 * mul_fact / 10, arg);
        int64_t remainder = arg % mul_fact;
        if (remainder == pivot_point)
        {
            int64_t val = (mul_fact - remainder) / mul_fact;
            return (val + (val & 1)) * mul_fact;
        }
        else
        {
            return ((arg + pivot_point) / mul_fact) * mul_fact;
        }
    }
}

double complex cpyc_expm1(double complex x) {
    double a = sin(cimag(x) * 0.5);
    return (expm1(creal(x)) * cos(cimag(x)) - 2. * a * a) + exp(creal(x)) * sin(cimag(x)) * _Complex_I;
}

extern inline double       pyc_radians(double degrees);
extern inline double       pyc_degrees(double radians);
extern inline int64_t      pyc_modulo(int64_t a, int64_t b);
extern inline double        pyc_fmodulo(double a, double b);
extern inline double complex complex_min(double complex a, double complex b);
extern inline double complex complex_max(double complex a, double complex b);
extern inline int8_t py_floor_div_int8_t(int8_t x, int8_t y);
extern inline int16_t py_floor_div_int16_t(int16_t x, int16_t y);
extern inline int32_t py_floor_div_int32_t(int32_t x, int32_t y);
extern inline int64_t py_floor_div_int64_t(int64_t x, int64_t y);
extern inline float complex py_csign_float_complex(float complex x);
extern inline double complex py_csign_double_complex(double complex x);
extern inline long double complex py_csign_long_double_complex(long double complex x);
