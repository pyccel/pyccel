# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

f1 = lambda x: x**2 + 1
f2 = lambda x,y: x**2 + f1(y)*f1(x)
g1 = lambda x: f1(x)**2 + 1

# lambda expressions can be printed

#$ header m1(double)
m1 = lambdify(g1)
print(f1)
print(f2)
print(g1)
