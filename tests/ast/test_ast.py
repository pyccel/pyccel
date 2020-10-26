# pylint: disable=missing-function-docstring, missing-module-docstring/
from sympy import symbols, MatrixSymbol, Matrix, Idx, Range
from sympy.utilities.pytest import raises


from pyccel.ast import (Assign, AugAssign, datatype,
                        Bool, Int, Float, Double, Void,
                        IndexedVariable,
                        For, Variable, FunctionDef)

n = Variable('int', 'n')
x = Variable('double', 'x')
y = Variable('double', 'y')

#A = MatrixSymbol('A', 3, 1)
#mat = Matrix([1, 2, 3])

B = IndexedVariable('B', dtype='double')
i = Variable('int', 'i')

def test_Assign():
    # Here we just do things to show they don't error
    Assign(x, y)
    Assign(x, 0)
#    Assign(A, mat)
#    Assign(A[1, 0], 0)
#    Assign(A[1, 0], x)
    Assign(B[i], x)
    Assign(B[i], 0)
    a = Assign(x, y)
    assert a.func(*a.args) == a
#    # Here we test things to show that they error
#    # Matrix to scalar
#    raises(ValueError, lambda: Assign(B[i], A))
#    raises(ValueError, lambda: Assign(B[i], mat))
#    raises(ValueError, lambda: Assign(x, mat))
#    raises(ValueError, lambda: Assign(x, A))
#    raises(ValueError, lambda: Assign(A[1, 0], mat))
#    # Scalar to matrix
#    raises(ValueError, lambda: Assign(A, x))
#    raises(ValueError, lambda: Assign(A, 0))
#    # Non-atomic lhs
#    raises(TypeError, lambda: Assign(mat, A))
#    raises(TypeError, lambda: Assign(0, x))
#    raises(TypeError, lambda: Assign(x * x, 1))
#    raises(TypeError, lambda: Assign(A + A, mat))
#    raises(TypeError, lambda: Assign(B, 0))


def test_AugAssign():
    # Here we just do things to show they don't error
    AugAssign(x, '+', y)
    AugAssign(x, '+', 0)
#    AugAssign(A, '+', mat)
#    AugAssign(A[1, 0], '+', 0)
#    AugAssign(A[1, 0], '+', x)
    AugAssign(B[i], '+', x)
    AugAssign(B[i], '+', 0)
    a = AugAssign(x, '+', y)
    assert a.func(*a.args) == a
#    # Here we test things to show that they error
#    # Matrix to scalar
#    raises(ValueError, lambda: AugAssign(B[i], '+', A))
#    raises(ValueError, lambda: AugAssign(B[i], '+', mat))
#    raises(ValueError, lambda: AugAssign(x, '+', mat))
#    raises(ValueError, lambda: AugAssign(x, '+', A))
#    raises(ValueError, lambda: AugAssign(A[1, 0], '+', mat))
#    # Scalar to matrix
#    raises(ValueError, lambda: AugAssign(A, '+', x))
#    raises(ValueError, lambda: AugAssign(A, '+', 0))
#    # Non-atomic lhs
#    raises(TypeError, lambda: AugAssign(mat, '+', A))
#    raises(TypeError, lambda: AugAssign(0, '+', x))
#    raises(TypeError, lambda: AugAssign(x * x, '+', 1))
#    raises(TypeError, lambda: AugAssign(A + A, '+', mat))
#    raises(TypeError, lambda: AugAssign(B, '+', 0))


def test_datatype():
    assert Bool is datatype('bool')
    assert Int is datatype('int')
    assert Float is datatype('float')
    assert Double is datatype('double')
    assert Void is datatype('void')
    # Check inferred types
    assert datatype(x) is Double
    assert datatype(n) is Int
    # This should work (I think), but doesn't due to how SymPy handles bools.
    # assert datatype(b) is Bool
#    assert datatype(A) is Double
#    assert datatype(mat) is Int
    d = datatype('int')
    assert d.func(*d.args) == d


#def test_For():
#    f = For(n, Range(0, 3), (Assign(A[n, 0], x + n), AugAssign(x, '+', y)))
#    f = For(n, (1, 2, 3, 4, 5), (Assign(A[n, 0], x + n),))
#    assert f.func(*f.args) == f
#    raises(TypeError, lambda: For(n, x, (x + y,)))


def test_Variable():
    v = Variable('int', 'v')
    assert v.func(*v.args) == v
    Variable('double', 'A')


##def test_Arguments():
##    a = InArgument('int', x)
##    b = OutArgument('double', x)
##    c = InOutArgument('float', A)
##    assert a.func(*a.args) == a
##    assert b.func(*b.args) == b
##    assert c.func(*c.args) == c
##
##
##def test_FunctionDef():
##    ax = InArgument('double', x)
##    ay = InArgument('double', y)
##    f = FunctionDef('test', (ax, ay), (Return(x + y),), (Result('double'),))
##    assert f.func(*f.args) == f
##    raises(TypeError, lambda: FunctionDef('test', (x, y), (Return(x + y),), (Result('double'),)))
##    raises(TypeError, lambda: FunctionDef('test', (ax, ay), (Return(x + y),), (x + y,)))
#
#
##def test_Return():
##    r = Return(x + y)
##    assert r.func(*r.args) == r

