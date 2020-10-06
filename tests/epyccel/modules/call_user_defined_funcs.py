# pylint: disable=missing-function-docstring, missing-module-docstring/
# This module test call user defined functions
# through nested functions

from pyccel.decorators import types

def do_nothing():
    x = 0
    x *= 0

@types('real')
def not_change(s):
    s *= s

@types('real', 'real')
def my_div(a, b):
    return a / b

@types('real', 'real')
def my_mult(a, b):
    return a * b

def my_pi():
    return 3.141592653589793

@types('real')
def my_cub(r):
    return r * r * r

@types('real')
def circle_volume(radius):
    do_nothing()
    volume = my_mult(my_mult(my_div(3. , 4.), my_pi()), my_cub(radius))
    not_change(volume)
    return volume
