# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy import array
from pyccel.decorators import elemental


@elemental
def square(x: float):
    s = x * x
    return s


if __name__ == "__main__":
    a = 2.0
    b = square(a)
    print(b)

    xs = array([1.0, 2.0, 3.0])
    ys = square(xs)
    print(ys)
