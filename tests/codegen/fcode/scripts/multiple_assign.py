# pylint: disable=missing-function-docstring, missing-module-docstring
#$ header g(double, double) results(double, double)
def g(x,v):
    m = x - v
    t =  2.0 * m
    z =  2.0 * t
    return t, z


z = 0.0
t = 0.0

# ...
xd = 5.
w = 5.
[z, t] = g(xd,w)

