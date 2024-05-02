# pylint: disable=missing-function-docstring, missing-module-docstring
# This module test call user defined functions
# through nested functions


def do_nothing():
    x = 0
    x *= 0

def not_change(s : 'float'):
    s *= s

def my_div(a : 'float', b : 'float'):
    return a / b

def my_mult(a : 'float', b : 'float'):
    return a * b

def my_pi():
    return 3.141592653589793

def my_cub(r : 'float'):
    return r * r * r

def circle_volume(radius : 'float'):
    do_nothing()
    volume = my_mult(my_mult(my_div(3. , 4.), my_pi()), my_cub(radius))
    not_change(volume)
    return volume

def arr_mult_scalar(T: 'int[:]', t: int = 13):
    x = T * t
    return x

def alias(T: 'int[:]', t: int):
    x = arr_mult_scalar(T, t=t)
    y = arr_mult_scalar(t=t, T=T)
    return x, y
