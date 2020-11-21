# pylint: disable=missing-function-docstring, missing-module-docstring/

#$ header function funct_c(int32[:], const int32[:])
def funct_c( x, y ):
    x[:] *= y


#$ header function f(int, int) &
#$                 results (int)

def f(a,b):
    c = a + b
    return c


#$ header function g(int, int) &
#$                 results &
#$ (int, int)

def g(a,b):
    c = a + b
    d = a - b
    return c, d

print('hello world')
