# pylint: disable=missing-class-docstring,  missing-function-docstring, missing-module-docstring

class Point:
    def __init__(self : 'Point'):
        pass

    def addition(self : 'Point', a : float, b : float):
        return a + b

    def subtraction(self : 'Point', a : 'float[:]', b : 'float[:]'):
        return a - b
