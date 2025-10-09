# Attempting to overwrite constant variable
# pylint: disable=missing-function-docstring, missing-module-docstring, missing-class-docstring

def f1():
    from numpy import pi
    pi = 4
    return pi

