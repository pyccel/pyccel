/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

#ifndef         PYC_MATH_C_H
#define         PYC_MATH_C_H

#include <stdint.h>

/*
** (N % M) + M and fmod(N, M) + M are used to handle the negative
** operands of modulo operator.
*/

#define MOD_PYC(N, M) \
    ({long int RES_MOD = (N % M); ((N < 0 ^ M < 0) && (RES_MOD ) != 0) ? (RES_MOD) + M : (RES_MOD);})
#define FMOD_PYC(N, M) \
    ({double RES_FMOD = fmod(N, M); ((N < 0 ^ M < 0) && RES_FMOD != 0  ? RES_FMOD + M : RES_FMOD);})


int64_t         pyc_factorial(int64_t n);
int64_t         pyc_gcd (int64_t a, int64_t b);
int64_t         pyc_lcm (int64_t a, int64_t b);
double          pyc_radians(double degrees);
double          pyc_degrees(double radians);

#endif
