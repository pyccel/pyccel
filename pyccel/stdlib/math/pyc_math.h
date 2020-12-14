#ifndef         PYC_MATH_H
#define         PYC_MATH_H

#include <stdint.h>

int64_t         pyc_factorial(int64_t n);
int64_t         pyc_gcd (int64_t a, int64_t b);
int64_t         pyc_lcm (int64_t a, int64_t b);
double          pyc_radians(double degrees);
double          pyc_degrees(double radians);

#endif
