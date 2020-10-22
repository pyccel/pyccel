# pylint: disable=missing-function-docstring, missing-module-docstring/
from numpy.random import randint
from numpy import equal

from pyccel.epyccel import epyccel
from modules import functionals

def compare_epyccel(f, *args):
    f2 = epyccel(f)
    out1 = f(*args)
    out2 = f2(*args)
    assert equal(out1, out2).all()

def test_functional_for_1d_range():
    compare_epyccel(functionals.functional_for_1d_range)

def test_functional_for_1d_var():
    y = randint(99,size = 4)
    compare_epyccel(functionals.functional_for_1d_var, y)

def test_functional_for_2d_range():
    compare_epyccel(functionals.functional_for_2d_range)

def test_functional_for_2d_var_range():
    y = randint(99, size = 3)
    compare_epyccel(functionals.functional_for_2d_var_range, y)

def test_functional_for_2d_var_var():
    y = randint(99, size = 3)
    z = randint(99, size = 2)
    compare_epyccel(functionals.functional_for_2d_var_var, y, z)

def test_functional_for_2d_dependant_range():
    compare_epyccel(functionals.functional_for_2d_dependant_range_1)
    compare_epyccel(functionals.functional_for_2d_dependant_range_2)
    compare_epyccel(functionals.functional_for_2d_dependant_range_3)

def test_functional_for_2d_array_range():
    idx = randint(28)
    compare_epyccel(functionals.functional_for_2d_array_range,idx)

def test_functional_for_3d_range():
    compare_epyccel(functionals.functional_for_3d_range)
