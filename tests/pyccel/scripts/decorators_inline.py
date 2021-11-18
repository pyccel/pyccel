# pylint: disable=missing-function-docstring, missing-module-docstring/
import numpy as np
from pyccel.decorators import inline

pi = 3.14159

@inline
def get_powers(s : int):
    return s, s*s, s*s*s

@inline
def power_4(s : int):
    tmp = s*s
    return tmp*tmp

@inline
def f(s : int):
    return power_4(s) / 2

@inline
def sin_base_1(d : float):
    return np.sin(2*pi*d)

if __name__ == '__main__':
    print(get_powers(3))
    a,b,c = get_powers(4)
    print(a,b,c)
    print(power_4(5))
    print(f(3))
    print(sin_base_1(0.5))
    print(sin_base_1(0.7))
