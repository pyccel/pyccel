/* --------------------------------------------------------------------------------------- */
/* This file is part of Pyccel which is released under MIT License. See the LICENSE file   */
/* or go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. */
/* --------------------------------------------------------------------------------------- */

#ifndef NUMPY_H
# define NUMPY_H

# include <complex.h>
# include <stdbool.h>
# include <stdint.h>
# include <math.h>

#define SIGN(x) (x ? (x < 0 ? -1 : 1) : 0)

long long int isign(long long int x);
double fsign(double x);
double complex csign(double complex x);

#endif
