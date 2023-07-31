# pylint: disable=missing-function-docstring, missing-module-docstring
import numpy as np

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

@inline
def fill_pi(a : 'float[:]'):
    pi = 3.14159
    for i in range(a.shape[0]):
        a[i] = pi

def not_inline():
    return 1.602e-19

@inline
def positron_charge():
    return -not_inline()

if __name__ == '__main__':
    print(get_powers(3))
    a,b,c = get_powers(4)
    print(a,b,c)
    print(power_4(5))
    print(f(3))
    print(sin_base_1(0.5))
    print(sin_base_1(0.7))
    arr = np.empty(4)
    fill_pi(arr)
    print(arr)
