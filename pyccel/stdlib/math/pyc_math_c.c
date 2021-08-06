/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

#include "pyc_math_c.h"
#include <math.h>

/*---------------------------------------------------------------------------*/
int64_t        pyc_factorial(int64_t n)
{
    int64_t    res = 1;

    /* ValueError: factorial() not defined for negative values */
    if (n < 0)
        return 0;
    for (int64_t i = 2; i <= n; i++)
        res *= i;
    return (res);
}
/*---------------------------------------------------------------------------*/
int64_t        pyc_gcd (int64_t a, int64_t b)
{
    while (b) {
        a %= b;
        /* swap a and b*/
        a = a ^ b;
        b = b ^ a;
        a = a ^ b;
    }
    return a;
}
/*---------------------------------------------------------------------------*/
int64_t        pyc_lcm (int64_t a, int64_t b)
{
    return a / pyc_gcd(a, b) * b;
}
/*---------------------------------------------------------------------------*/
double      pyc_radians(double degrees)
{
    return (degrees * (M_PI / 180));
}
/*---------------------------------------------------------------------------*/
double      pyc_degrees(double radians)
{
    return radians * (180.0 / M_PI);
}
/*---------------------------------------------------------------------------*/
