# pylint: disable=missing-function-docstring, missing-module-docstring/
from numpy.random import randint
from numpy import isclose

from pyccel.epyccel import epyccel
from pyccel.decorators import types

RTOL = 2e-14
ATOL = 1e-15

# -------------------- normal cases round ---------------------- #

@types('float', 'int')
def roundNdigits(x, n):
    return round(x, n)

@types('float')
def roundInt(x):
    return round(x)

def testIntPositive():
    f = epyccel(roundInt, language='c')

    x = randint(1, 1000) / 100

    f_output = f(x)

    python_output = roundInt(x)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    f = epyccel(roundInt, language='c')

    # round up

    x = 3.678

    f_output = f(x)

    python_output = roundInt(x)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    # round down

    x = 3.431

    f_output = f(x)

    python_output = roundInt(x)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    # round mid

    x = 3.5

    f_output = f(x)

    python_output = roundInt(x)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))


def testIntNegative():
    f = epyccel(roundInt, language='c')

    x = randint(-1000, -1) / 100

    f_output = f(x)

    python_output = roundInt(x)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    # round up

    x = -3.678

    f_output = f(x)

    python_output = roundInt(x)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    # round down

    x = -3.431

    f_output = f(x)

    python_output = roundInt(x)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    # round mid

    x = -3.5

    f_output = f(x)

    python_output = roundInt(x)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

def testNdigitsPositive():
    f = epyccel(roundNdigits, language='c')

    x = randint(1, 1000) / 100

    n = randint(1, 1000)

    print("testNdigitsPositive")

    print(x, n)

    f_output = f(x, n)

    python_output = roundNdigits(x, n)

    print(f_output, python_output)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    # round up

    x = 3.678

    n = 2

    f_output = f(x, n)

    python_output = roundNdigits(x, n)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    # round down

    x = 3.431

    n = 2

    print("testNdigitsPositive round down")

    print(x, n)

    f_output = f(x, n)

    python_output = roundNdigits(x, n)

    print(f_output, python_output)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    # negative ndigits

    x = 355.431

    n = -2

    print("testNdigitsPositive negative ndigits")

    print(x, n)

    f_output = f(x, n)

    python_output = roundNdigits(x, n)

    print(f_output, python_output)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    # negative ndigits mid

    x = 15.0

    n = -1

    print("testNdigitsPositive negative ndigits mid")

    print(x, n)

    f_output = f(x, n)

    python_output = roundNdigits(x, n)

    print(f_output, python_output)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    # round mid

    x = 3.5

    n = 0

    print("testNdigitsPositive round mid")

    print(x, n)

    f_output = f(x, n)

    python_output = roundNdigits(x, n)

    print(f_output, python_output)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

def testNdigitsNegative():
    f = epyccel(roundNdigits, language='c')

    x = randint(-1000, -1) / 100

    n = randint(-1000, -1)

    print("testNdigitsNegative")

    print(x, n)

    f_output = f(x, n)

    python_output = roundNdigits(x, n)

    print(f_output, python_output)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    # round up

    x = -3.678

    n = 2

    print("testNdigitsNegative round up")

    print(x, n)

    f_output = f(x, n)

    python_output = roundNdigits(x, n)

    print(f_output, python_output)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    # round down

    x = -3.431

    n = 2

    print("testNdigitsNegative round down")

    print(x, n)

    f_output = f(x, n)

    python_output = roundNdigits(x, n)

    print(f_output, python_output)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    # negative ndigits

    x = -355.431

    n = -2

    print("testNdigitsNegative negative ndigits")

    print(x, n)

    f_output = f(x, n)

    python_output = roundNdigits(x, n)

    print(f_output, python_output)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    # negative ndigits mid

    x = -15.0

    n = -1

    print("testNdigitsNegativ negative ndigits mid")

    print(x, n)

    f_output = f(x, n)

    python_output = roundNdigits(x, n)

    print(f_output, python_output)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

    # round mid

    x = -3.5

    n = 0

    print("testNdigitsNegative round mid")

    print(x, n)

    f_output = f(x, n)

    python_output = roundNdigits(x, n)

    print(f_output, python_output)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))

def testNdigitsEdgeCases():
    f = epyccel(roundNdigits, language='c')

    x = 2.675

    n = 2

    print("testNdigitsEdgeCases 2")

    print(x, n)

    f_output = f(x, n)

    python_output = roundNdigits(x, n)

    print(f_output, python_output)

    assert isclose(f_output, python_output, rtol=RTOL, atol=ATOL)

    assert isinstance(f_output ,type(python_output))


if __name__ == '__main__':
    testIntPositive()
    testIntNegative()
    testNdigitsPositive()
    testNdigitsNegative()
    testNdigitsEdgeCases()
