/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

#ifndef         PYC_MATH_C_H
#define         PYC_MATH_C_H
#include <math.h>
#include <stdint.h>

/*
** (N % M) + M and fmod(N, M) + M are used to handle the negative
** operands of modulo operator.
*/

int64_t             pyc_factorial(int64_t n);
int64_t             pyc_gcd (int64_t a, int64_t b);
int64_t             pyc_lcm (int64_t a, int64_t b);

inline double       pyc_radians(double degrees)
{
    return (degrees * (M_PI / 180));
}
inline double       pyc_degrees(double radians)
{
    return radians * (180.0 / M_PI);
}
inline int64_t      pyc_modulo(int64_t a, int64_t b){
        int64_t res_mod = a % b;
        if(!(a < 0 ^ b < 0) || res_mod == 0 )
            return(res_mod);
        else
            return(res_mod + b);
}
inline double        pyc_fmodulo(double a, double b){
        double res_mod = fmod(a, b);
        if(!(a < 0 ^ b < 0) || res_mod == 0 )
            return(res_mod);
        else
            return(res_mod + b);
}
#endif
