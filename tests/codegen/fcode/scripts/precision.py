# pylint: disable=missing-function-docstring, missing-module-docstring/
# pylint: disable=unused-variable

#$ header function incr_(int*8)
def incr_(x):
    #$ header function decr_(int*8)
    def decr_(y):
        y = y-1
    x = x + 1

#$ header function f1(int*8, int*16, int*4) results(int*16)
def f1(x, n=2, m=3):
    y = x - n*m
    return y

#$ header function f2(double*8, int*4) results(double*4)
def f2(x, m=None):
    if m is None:
        y = x + 1
    else:
        y = x - 1
    return y


from numpy import float32, float64 , int32, int64, int as np_int, float as np_float, complex as np_complex, complex64, complex128

x1 = np_int(6)
x2 = int32(6)
x3 = int64(6)
y1 = np_float(6)
y2 = float32(6)
y3 = float64(6)
z1 = np_complex(6)
z2 = complex64(6)
z3 = complex128(6)
