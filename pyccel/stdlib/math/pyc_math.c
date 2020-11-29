#include "pyc_math.h"
#include <math.h>

/*---------------------------------------------------------------------------*/
long        pyc_factorial(long n)
{
    long    res = 1;

    /* ValueError: factorial() not defined for negative values */
    if (n < 0)
        return 0;
    for (int i = 2; i <= n; i++)
        res *= i;
    return (res);
}
/*---------------------------------------------------------------------------*/
long        pyc_gcd (long a, long b)
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
long        pyc_lcm (long a, long b)
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
