from sympy import (symbols, sin, cos, Dict, sqrt, tan, simplify, MatrixSymbol,
        MatrixExpr, Matrix)
from sympy.utilities.pytest import raises
from sympy.core.assumptions import _assume_defined

from pyccel.types.ast import Assign, InArgument, OutArgument, datatype, Double
from pyccel.types.routines import (RoutineReturn, RoutineInplace, Routine,
        routine, routine_result, ScalarRoutineCallResult, MatrixRoutineCallResult)


a, b, c = symbols('a, b, c')
out = symbols('out')
a_arg = InArgument(Double, a)
b_arg = InArgument(Double, b)
c_arg = InArgument(Double, c)
out_arg = OutArgument(Double, out)
expr = sin(a) + cos(b**2)*c
inp_expr = Assign(out, expr)

x = MatrixSymbol('x', 3, 1)
x_arg = OutArgument(Double, x)
matres = Matrix([sin(a), cos(b), c])
mat_expr = Assign(x, matres)


def test_arg_invariance():
    r = RoutineReturn(Double, expr)
    assert r.func(*r.args) == r
    rip = RoutineInplace(out_arg, expr)
    assert rip.func(*rip.args) == rip
    rout = Routine('test', [a_arg, b_arg, c_arg, out_arg], [r, rip])
    assert rout.func(*rout.args) == rout
    rcall = rout(a, b, c, out)
    assert rcall.func(*rcall.args) == rcall
    rcallret = rcall.returns
    assert rcallret.func(*rcallret.args) == rcallret
    rcallinp = rcall.inplace[out]
    assert rcallinp.func(*rcallinp.args) == rcallinp


def test_routine_result():
    r = routine_result(expr)
    assert r == RoutineReturn(Double, expr)
    r = routine_result(Assign(out, expr))
    assert r == RoutineInplace(out_arg, expr)


def test_routine():
    test = routine('test', (a, b, c), expr)
    assert test == Routine('test', (a_arg, b_arg, c_arg), (RoutineReturn(Double, expr),))
    test = routine('test', (a, b, c, out), inp_expr)
    assert test == Routine('test', (a_arg, b_arg, c_arg, out_arg), (RoutineInplace(out_arg, expr),))
    test = routine('test', (a, b, c, x), mat_expr)
    assert test == Routine('test', (a_arg, b_arg, c_arg, x_arg), (RoutineInplace(x_arg, matres),))
    # Test arg errors
    raises(ValueError, lambda: routine('test', (a, b, c), inp_expr))
    raises(ValueError, lambda: routine('test', (a, b, c, out), expr))


def test_Routine():
    test = routine('test', (a, b, c), expr)
    assert test.name == symbols('test')
    assert test.arguments == (a_arg, b_arg, c_arg)
    assert test.results == (routine_result(expr),)
    assert test.returns == (routine_result(expr),)
    assert test.inplace == ()
    # Multiple results
    test = routine('test', (a, b, c, out), (expr, inp_expr))
    assert test.arguments == (a_arg, b_arg, c_arg, out_arg)
    assert test.results == (routine_result(expr), routine_result(inp_expr))
    assert test.returns == (routine_result(expr),)
    assert test.inplace == (routine_result(inp_expr),)
    # Matrix result
    test = routine('test', (a, b, c, x), (expr, mat_expr))
    assert test.arguments == (a_arg, b_arg, c_arg, x_arg)
    assert test.results == (routine_result(expr), routine_result(mat_expr))
    assert test.returns == (routine_result(expr),)
    assert test.inplace == (routine_result(mat_expr),)


def test_RoutineCall():
    test = routine('test', (a, b, c), expr)
    rcall = test(1, 2, 3)
    assert rcall.routine == test
    assert rcall.arguments == (1, 2, 3)
    assert rcall.returns == ScalarRoutineCallResult(rcall, -1)
    assert rcall.inplace == Dict()
    # Multiple results
    test = routine('test', (a, b, c, out), (expr, inp_expr))
    rcall = test(1, 2, 3, out)
    assert rcall.arguments == (1, 2, 3, out)
    assert rcall.returns == ScalarRoutineCallResult(rcall, -1)
    assert rcall.inplace == Dict({out: ScalarRoutineCallResult(rcall, out)})
    # Matrix result
    test = routine('test', (a, b, c, x), (expr, mat_expr))
    rcall = test(1, 2, 3, x)
    assert rcall.arguments == (1, 2, 3, x)
    assert rcall.returns == ScalarRoutineCallResult(rcall, -1)
    assert rcall.inplace == Dict({x: MatrixRoutineCallResult(rcall, x)})


def test_ScalarRoutineCallResult():
    test = routine('test', (a, b, c, out), (expr, inp_expr))
    rcall = test(1, 2, 3, out)
    res_expr = expr.subs({a: 1, b: 2, c: 3})
    ret = rcall.returns
    inp = rcall.inplace[out]
    assert ret.expr == res_expr
    assert inp.expr == res_expr
    assert ret.free_symbols == set([out])
    assert inp.free_symbols == set([out])


def test_ScalarRoutineCallResult_assumptions():
    test = routine('test', (a, b, c), expr)
    rcall = test(1, 2, 3)
    res_expr = expr.subs({a: 1, b: 2, c: 3})
    ret = rcall.returns
    # Use results in some expressions (will error if fails)
    ret*a + sin(b*c)
    sin(ret)
    sqrt(ret)*ret
    def assump_checker(ret, expr, name):
        name = 'is_' + name
        assert getattr(ret, name) == getattr(ret, name)
    # Check the aliasing of assumptions
    for name in _assume_defined:
        assump_checker(ret, res_expr, name)
    # See if the assumption checks broke anything
    ret*a + sin(b*c)
    sin(ret)
    sqrt(ret)*ret


def test_ScalarRoutineCallResult_subs():
    test = routine('test', (a, b, c), expr)
    rcall = test(1, 2, 3)
    ret = rcall.returns
    assert ret.subs(a, 1) == ret
    assert ret.subs(1, a) == test(a, 2, 3).returns


def test_ScalarRoutineCallResult_simplify():
    ret = routine('test', (a, b, c), expr)(1, 2, 3).returns
    test_expr = ret * sin(a)/cos(a)
    assert simplify(test_expr) == ret * tan(a)


def test_MatrixRoutineCallResult():
    test = routine('test', (a, b, c, x), mat_expr)
    rcall = test(1, 2, 3, x)
    matres_expr = matres.subs({a: 1, b: 2, c: 3})
    inp = rcall.inplace[x]
    assert inp.expr == matres_expr
    assert inp.free_symbols == set([x])


def test_MatrixRoutineCallResult_assumptions():
    test = routine('test', (a, b, c, x), mat_expr)
    rcall = test(1, 2, 3, x)
    matres_expr = matres.subs({a: 1, b: 2, c: 3})
    ret = rcall.inplace[x]
    # Use results in some expressions (will error if fails)
    ret + x
    (ret*ret.T).inverse()
    ret[1]
    ret[1:2]
    def assump_checker(ret, expr, name):
        name = 'is_' + name
        assert getattr(ret, name) == getattr(ret, name)
    # Check the aliasing of assumptions
    for name in _assume_defined:
        assump_checker(ret, matres_expr, name)
    # See if the assumption checks broke anything
    ret + x
    (ret*ret.T).inverse()
    ret[1]
    ret[1:2]


def test_MatrixRoutineCallResult_subs():
    test = routine('test', (a, b, c, x), mat_expr)
    rcall = test(1, 2, 3, x)
    ret = rcall.inplace[x]
    assert ret.subs(a, 1) == ret
    ret = test(a, a, a, x).inplace[x]
    assert ret.subs(a, b) == test(b, b, b, x).inplace[x]


def test_ScalarRoutineCallResult_simplify():
    ret = routine('test', (a, b, c, x), mat_expr)(1, 2, 3, x).inplace[x]
    test_expr = ret*ret.T + x*x.T
    assert simplify(test_expr) == test_expr
