# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import sympy

#$ header function g(double)
@sympy
def f(x : float):
    from sympy import tan
    return tan(x).diff()

print(f)

g = lambdify(f)
print(g(3.0))
