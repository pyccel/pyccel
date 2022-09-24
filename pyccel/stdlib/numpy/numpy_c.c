#include "numpy_c.h"

long int isign(long int x)
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