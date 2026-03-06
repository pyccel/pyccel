/* -------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file  */
/* or go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details. */
/* -------------------------------------------------------------------------------------- */

#include "pyc_math_cpp.hpp"

/*---------------------------------------------------------------------------*/
int64_t pyc_gcd(int64_t a, int64_t b)
{
    a = a > 0 ? a : -a;
    b = b > 0 ? b : -b;
    while (b)
    {
        a %= b;
        std::swap(a, b);
    }
    return a;
}
/*---------------------------------------------------------------------------*/
int64_t pyc_lcm(int64_t a, int64_t b)
{
    a = std::copysign(a, 1);
    b = std::copysign(b, 1);
    return a / pyc_gcd(a, b) * b;
}

/*---------------------------------------------------------------------------*/

double pyc_bankers_round(double arg, int64_t ndigits)
{
    double factor = std::pow(10.0, ndigits);
    arg *= factor;

    double nearest_int_fix = std::copysign(0.5, arg);

    double rnd = (int64_t)(arg + nearest_int_fix);

    double diff = arg - rnd;

    if (ndigits <= 0 && (diff == 0.5 || diff == -0.5))
    {
        rnd = ((int64_t)(arg * 0.5 + nearest_int_fix)) * 2.0;
    }

    return rnd / factor;
}

int64_t pyc_bankers_round(int64_t arg, int64_t ndigits)
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

        int64_t pivot_point = std::copysign(5 * mul_fact / 10, arg);
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

extern inline double       pyc_radians(double degrees);
extern inline double       pyc_degrees(double radians);
