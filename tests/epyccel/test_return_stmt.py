# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types
from pyccel.epyccel import epyccel

def test_single_return_var_assign(language):
    def single_return_var_assign():
        y = 3
        return y
    epyc_single_return_var_assign = epyccel(single_return_var_assign, language=language)
    assert (epyc_single_return_var_assign() == single_return_var_assign())

def test_assign_vars_return(language):
    @types('int', 'int')
    def assign_vars_return(a,b):
        c = a+b
        d = a-b
        return c+d
    epyc_assign_vars_return = epyccel(assign_vars_return, language=language, fflags="-Werror -Wunused-variable")
    assert (epyc_assign_vars_return(3, 4) == assign_vars_return(3, 4))


def test_sum_in_single_return(language):
    @types('int', 'int')
    def sum_in_single_return(a,b):
        c = a + b
        return c
    epyc_sum_in_single_return = epyccel(sum_in_single_return, language=language, fflags="-Werror -Wunused-variable")
    assert (epyc_sum_in_single_return(7, 2) == sum_in_single_return(7, 2))

def test_return_expr(language):
    @types('int', 'int')
    def return_expr(x, y):
        return x + y
    epyc_return_expr = epyccel(return_expr, language=language, fflags="-Werror -Wunused-variable")
    assert (epyc_return_expr(7, 2) == return_expr(7, 2))

def test_return_single_var(language):
    @types('int')
    def return_single_var(x):
        return x
    epyc_return_single_var = epyccel(return_single_var, language=language, fflags="-Werror -Wunused-variable")
    assert (epyc_return_single_var(7) == return_single_var(7))

def test_return_scalare(language):
    def return_scalare():
        return 5
    epyc_return_scalare = epyccel(return_scalare, language=language, fflags="-Werror -Wunused-variable")
    assert (epyc_return_scalare() == return_scalare())

def test_multi_return_scalare(language):
    def multi_return_scalare():
        return 5, 7
    epyc_multi_return_scalare = epyccel(multi_return_scalare, language=language, fflags="-Werror -Wunused-variable")
    assert (epyc_multi_return_scalare() == multi_return_scalare())

def test_multi_return_vars(language):
    @types('int', 'int')
    def multi_return_vars(a, b):
        return a, b
    epyc_multi_return_vars = epyccel(multi_return_vars, language=language, fflags="-Werror -Wunused-variable")
    assert (epyc_multi_return_vars(7, 2) == multi_return_vars(7, 2))

def test_multi_return_vars_expr(language):
    @types('int', 'int')
    def multi_return_vars_expr(a, b):
        return (a-b), (a+b)
    epyc_multi_return_vars_expr = epyccel(multi_return_vars_expr, language=language, fflags="-Werror -Wunused-variable")
    assert (epyc_multi_return_vars_expr(7, 2) == multi_return_vars_expr(7, 2))

def test_scalare_multi_return_stmts(language):
    @types('int')
    def scalare_multi_return_stmts(a):
        a = 7
        if a:
            return 1
        else:
            return 2
        a = 4
        return a
    epyc_scalare_multi_return_stmts = epyccel(scalare_multi_return_stmts, language=language, fflags="-Werror -Wunused-variable")
    assert (epyc_scalare_multi_return_stmts(7) == scalare_multi_return_stmts(7))
