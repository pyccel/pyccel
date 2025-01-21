# pylint: disable=missing-function-docstring, missing-module-docstring

def y(x : 'double'):
    from numpy import cos, sin
    return sin(x)**2 + (x**3 + x**2 - x - 1)/(x**2 + 2*x + 1) + cos(x)**2

if __name__ == '__main__':
    print(y(2.78))
    print(y(0.07))
