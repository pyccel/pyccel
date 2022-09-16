from numpy.random import randint
from numpy import isclose

from pyccel.epyccel import epyccel
from pyccel.decorators import types

RTOL = 2e-14
ATOL = 1e-09

# -------------------- normal cases round ---------------------- #


def comparisonTest():
    @types('float', 'int')
    def round1(x, n):
        return round(x, n)

    f = epyccel(round1, language='c')

    for i in range(100000):
        x = randint(9999)
        y = randint(1, 9999)
        z = randint(-99, 99)

        fraction = x/y

        print("fraction ==", fraction)
        print("ndigits ==", z)
        print("Pythround(fraction) ==", round1(fraction, z))
        print("Pyccround(fraction) ==", f(fraction, z))
        assert isclose(f(fraction, z), round1(
            fraction, z), rtol=RTOL, atol=ATOL)


if __name__ == '__main__':
    comparisonTest()
