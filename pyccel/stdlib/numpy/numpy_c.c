/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

#include "numpy_c.h"

/* numpy.sign for float, double and integers */
long long int isign(long long int x)
{
    return SIGN(x);
}

/* numpy.sign for float, double and integers */
double  fsign(double x)
{
    return SIGN(x);
}

/* numpy.sign for complex */
double complex csign(double complex x)
{
    return x ? ((!creal(x) && cimag(x) < 0) || (creal(x) < 0) ? -1 : 1) : 0;
}
