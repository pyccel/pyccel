from pyccel.decorators import pure, types
from pyccel.epyccel import epyccel

#==============================================================================
def compare_epyccel(f1, *args):
    f2 = epyccel(f1)
    out1 = f1(*args)
    out2 = f2(*args)
    assert all(r1==r2 for r1, r2 in zip(out1, out2))

#==============================================================================
def test_const_int_float():

    @pure
    def const_int_float():
        return 1, 3.4

    compare_epyccel(const_int_float)

# ...
def test_const_complex_bool_int():

    @pure
    def const_complex_bool_int():
        return 1+2j, False, 8

    compare_epyccel(const_complex_bool_int)

# ...
def test_expr_float_int_bool():

    @pure
    @types('int')
    def expr_complex_int_bool(n):
        return 0.5+n*1j, 2*n, n==3

    compare_epyccel(expr_complex_int_bool, 3)
