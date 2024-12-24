/* -------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file  */
/* or go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details. */
/* -------------------------------------------------------------------------------------- */

#include "pyc_math_c.h"

/*---------------------------------------------------------------------------*/
int64_t                     pyc_factorial(int64_t n)
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
int64_t                     pyc_gcd (int64_t a, int64_t b)
{
    a = a > 0 ? a : - a;
    b = b > 0 ? b : - b;
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
int64_t                     pyc_lcm (int64_t a, int64_t b)
{
    a = a > 0 ? a : - a;
    b = b > 0 ? b : - b;
    return a / pyc_gcd(a, b) * b;
}
/*---------------------------------------------------------------------------*/

/* numpy.sign for float, double and integers */
long long int isign(long long int x)
{
    return (x>0) - (x<0);
}

/* numpy.sign for float, double and integers */
double  fsign(double x)
{
    return (double)((x>0) - (x<0));
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

