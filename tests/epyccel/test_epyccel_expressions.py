# pylint: disable=missing-function-docstring, missing-module-docstring
from numpy.random import randint, rand
import numpy as np
from numpy import iinfo
from pyccel import epyccel

# Use int32 for Windows compatibility
min_int = iinfo(np.int32).min
max_int = iinfo(np.int32).max

def test_swap_basic(language):
    def swp(a : int, b : int):
        (a, b) = (b, a)
        return a, b

    f = epyccel(swp, language=language)
    assert f(2,4) == swp(2,4)
    assert f(-2,4) == swp(-2,4)
    assert f(4,100) == swp(4,100)
    x = randint(min_int, max_int)
    y = randint(min_int, max_int)
    assert f(x,y) == swp(x,y)

def test_swap_basic_2(language):
    def swp(a : int, b : int):
        a, b = b, a
        return a, b

    f = epyccel(swp, language=language)
    assert f(2,4) == swp(2,4)
    assert f(-2,4) == swp(-2,4)
    assert f(4,100) == swp(4,100)
    x = randint(min_int, max_int)
    y = randint(min_int, max_int)
    assert f(x,y) == swp(x,y)

def test_swap_basic_3(language):
    def swp(a : int, b : int, c : int):
        a, b, c = b, c, a
        return a, b, c

    f = epyccel(swp, language=language)
    assert f(2,4,8) == swp(2,4,8)
    assert f(-2,4,-6) == swp(-2,4,-6)
    assert f(4,100,234) == swp(4,100,234)
    x = randint(min_int, max_int)
    y = randint(min_int, max_int)
    z = randint(min_int, max_int)
    assert f(x,y,z) == swp(x,y,z)

def test_swap_basic_4(language):
    def swp(a : int, b : int, c : int):
        a, b, c = c, b, a #pylint: disable=self-assigning-variable
        return a, b, c

    f = epyccel(swp, language=language)
    assert f(2,4,8) == swp(2,4,8)
    assert f(-2,4,-6) == swp(-2,4,-6)
    assert f(4,100,234) == swp(4,100,234)
    x = randint(min_int, max_int)
    y = randint(min_int, max_int)
    z = randint(min_int, max_int)
    assert f(x,y,z) == swp(x,y,z)

def test_swap_index_1(language):
    def swp(a : int, b : int, c : int):
        l = [a,b,c]
        l[0], l[1] = l[1], l[0]
        return l[0], l[1], l[2]

    f = epyccel(swp, language=language)
    assert f(2,4,8) == swp(2,4,8)
    assert f(-2,4,-6) == swp(-2,4,-6)
    assert f(4,100,234) == swp(4,100,234)
    x = randint(min_int, max_int)
    y = randint(min_int, max_int)
    z = randint(min_int, max_int)
    assert f(x,y,z) == swp(x,y,z)

def test_swap_index_2(language):
    def swp(i : int, j : int):
        l = [1,2,3]
        l[i], l[j] = l[j], l[i]
        return l[0], l[1], l[2]

    f = epyccel(swp, language=language)
    assert f(0,1) == swp(0,1)
    assert f(1,0) == swp(1,0)
    assert f(2,1) == swp(2,1)

def test_multi_level_swap(language):
    def swp(a : int, b : int, c : int):
        d, (b, c) = a, (c, b)
        return a, b, c, d

    f = epyccel(swp, language=language)
    assert f(2,4,8) == swp(2,4,8)
    assert f(-2,4,-6) == swp(-2,4,-6)
    assert f(4,100,234) == swp(4,100,234)
    x = randint(min_int, max_int)
    y = randint(min_int, max_int)
    z = randint(min_int, max_int)
    assert f(x,y,z) == swp(x,y,z)

def test_multi_type_swap(language):
    def swp(a : float, b : int, c : float, d : int):
        a, b, c, d = c, d, a, b
        return a, b, c, d

    b = randint(min_int, max_int)
    d = randint(min_int, max_int)
    a = rand()*100
    c = rand()*100

    f = epyccel(swp, language=language)
    assert f(a,b,c,d) == swp(a,b,c,d)
    assert f(-2.,4,-6.,10) == swp(-2.,4,-6.,10)

def test_tuple_assign(language):
    def tup_assign(a : int, b : int):
        c, d = a, a+b
        return c, d

    f = epyccel(tup_assign, language=language)
    assert f(2,4) == tup_assign(2,4)
    assert f(-2,4) == tup_assign(-2,4)
    assert f(4,100) == tup_assign(4,100)
    x = randint(min_int//2, max_int//2)
    y = randint(min_int//2, max_int//2)
    assert f(x,y) == tup_assign(x,y)

def test_tuple_assign2(language):
    def tup_assign(a : int, b : int):
        a, d = a, a+b #pylint: disable=self-assigning-variable
        return a, b, d

    f = epyccel(tup_assign, language=language)
    assert f(2,4) == tup_assign(2,4)
    assert f(-2,4) == tup_assign(-2,4)
    assert f(4,100) == tup_assign(4,100)
    x = randint(min_int//2, max_int//2)
    y = randint(min_int//2, max_int//2)
    assert f(x,y) == tup_assign(x,y)

def test_tuple_assign3(language):
    def tup_assign(a : int):
        a, a = a+3, a+5 #pylint: disable=redeclared-assigned-name
        return a

    f = epyccel(tup_assign, language=language)
    assert f(2) == tup_assign(2)
    assert f(-2) == tup_assign(-2)
    assert f(40) == tup_assign(40)
    x = randint(min_int, max_int-5)
    assert f(x) == tup_assign(x)

