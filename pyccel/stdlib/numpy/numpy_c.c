#include "numpy_c.h"

/* numpy.sign for float, double and integers */
double  sign(double x)
{
    return x ? (x < 0 ? -1 : 1) : 0;
}

/* numpy.sign for complex */
double complex csign(double complex x)
{
    return x ? ((!creal(x) && cimag(x) < 0) || (creal(x) < 0) ? -1 : 1) : 0;
}