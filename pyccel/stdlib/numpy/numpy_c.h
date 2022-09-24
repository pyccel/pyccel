#ifndef NUMPY_H
# define NUMPY_H

# include <complex.h>
# include <stdbool.h>
# include <stdint.h>
# include <math.h>

#define SIGN(x) (x ? (x < 0 ? -1 : 1): 0)

long int isign(long int x);
double fsign(double x);
double complex csign(double complex x);

#endif