# coding: utf-8

# TODO: - raise error if the weak form is not linear/bilinear on the test/trial

from sympy.core.expr import Expr
from sympy.core.containers import Tuple
from sympy import Integer, Float


from sympy import symbols, Tuple, Lambda, Symbol, sympify, expand
from sympy import Add, Mul
from sympy import preorder_traversal, Expr
from sympy import Indexed, IndexedBase
from sympy import simplify
from sympy import S
from sympy.core.compatibility import is_sequence
from sympy import Basic
from sympy import Function

# ...
class LinearOperator(Function):
    """

    Examples
    ========

    """

    nargs = None
    name = 'Grad'

    def __new__(cls, *args, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        _coeffs_registery = (Integer, Float)

        expr = _args[0]
        if isinstance(expr, Add):
            args = expr.args
            args = [cls.eval(a) for a in expr.args]
            return Add(*args)

        if isinstance(expr, Mul):
            coeffs  = [a for a in expr.args if isinstance(a, _coeffs_registery)]
            vectors = [a for a in expr.args if not(a in coeffs)]

            a = S.One
            if coeffs:
                a = Mul(*coeffs)

            b = S.One
            if vectors:
                b = cls(Mul(*vectors), evaluate=False)

            return Mul(a, b)

        return cls(expr, evaluate=False)
# ...

# ...
class DifferentialOperator(LinearOperator):
    """
    This class is a linear operator that applies the Leibniz formula

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        _coeffs_registery = (Integer, Float)

        expr = _args[0]
        if isinstance(expr, Add):
            args = expr.args
            args = [cls.eval(a) for a in expr.args]
            return Add(*args)

        if isinstance(expr, Mul):
            coeffs  = [a for a in expr.args if isinstance(a, _coeffs_registery)]
            vectors = [a for a in expr.args if not(a in coeffs)]

            c = S.One
            if coeffs:
                c = Mul(*coeffs)

            V = S.One
            if vectors:
                if len(vectors) == 1:
                    V = cls(Mul(vectors[0]), evaluate=False)

                elif len(vectors) == 2:
                    a = vectors[0]
                    b = vectors[1]

                    fa = cls(a, evaluate=False)
                    fb = cls(b, evaluate=False)

                    V = a * fb + fa * b

                else:
                    V = cls(Mul(*vectors), evaluate=False)

            return Mul(c, V)

        return cls(expr, evaluate=False)
# ...

# ...
class dx(DifferentialOperator):
    pass
# ...

# ...
class dy(DifferentialOperator):
    pass
# ...

# ...
class dz(DifferentialOperator):
    pass
# ...

# ...
class DotBasic(Function):
    """

    Examples
    ========

    """

    nargs = None
    name = 'Dot'

    def __new__(cls, *args, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

class Dot_1d(DotBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]
        v = _args[1]

        return u * v

class Dot_2d(DotBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]
        v = _args[1]

        return u[0]*v[0] + u[1]*v[1]

class Dot_3d(DotBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]
        v = _args[1]

        return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
# ...

# ...
class CrossBasic(Function):
    """

    Examples
    ========

    """

    nargs = None
    name = 'Cross'

    def __new__(cls, *args, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

class Cross_2d(CrossBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]
        v = _args[1]

        return u[0]*v[1] - u[1]*v[0]

class Cross_3d(CrossBasic):
    """

    Examples
    ========

    """

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]
        v = _args[1]

        return Tuple(u[1]*v[2] - u[2]*v[1],
                     u[2]*v[0] - u[0]*v[2],
                     u[0]*v[1] - u[1]*v[0])
# ...


# ...
class GradBasic(Function):
    """

    Examples
    ========

    """

    nargs = None
    name = 'Grad'

    def __new__(cls, *args, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

class Grad_1d(GradBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return dx(u)

class Grad_2d(GradBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return Tuple(dx(u), dy(u))

class Grad_3d(GradBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return Tuple(dx(u), dy(u), dz(u))
# ...


# ...
class CurlBasic(Function):
    """

    Examples
    ========

    """

    nargs = None
    name = 'Curl'

    def __new__(cls, *args, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

class Curl_2d(CurlBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return Tuple( dy(u),
                     -dx(u))

class Curl_3d(CurlBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return Tuple(dy(u[2]) - dz(u[1]),
                     dz(u[0]) - dx(u[2]),
                     dx(u[1]) - dy(u[0]))
# ...

# ...
class Rot_2d(Function):
    """

    Examples
    ========

    """

    nargs = None
    name = 'Grad'

    def __new__(cls, *args, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return dy(u[0]) - dx(u[1])
# ...

# ...
class DivBasic(Function):
    """

    Examples
    ========

    """

    nargs = None
    name = 'Div'

    def __new__(cls, *args, **options):
        # (Try to) sympify args first

        if options.pop('evaluate', True):
            r = cls.eval(*args)
        else:
            r = None

        if r is None:
            return Basic.__new__(cls, *args, **options)
        else:
            return r

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

class Div_1d(DivBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return dx(u)

class Div_2d(DivBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return dx(u[0]) + dy(u[1])

class Div_3d(DivBasic):
    """

    Examples
    ========

    """

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        u = _args[0]

        return dx(u[0]) + dy(u[1]) + dz(u[2])
# ...

# ...
_operators_1d = [Dot_1d,
                 Grad_1d, Div_1d]

_operators_2d = [Dot_2d, Cross_2d,
                 Grad_2d, Curl_2d, Rot_2d, Div_2d]

_operators_3d = [Dot_3d, Cross_3d,
                 Grad_3d, Curl_3d, Div_3d]
# ...


# ... generic operators
class GenericFunction(Function):

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            return Indexed(self, *indices, **kw_args)
        else:
            return Indexed(self, indices, **kw_args)

class Dot(GenericFunction):
    pass

class Cross(GenericFunction):
    pass

class Grad(GenericFunction):
    pass

class Curl(GenericFunction):
    pass

class Rot(GenericFunction):
    pass

class Div(GenericFunction):
    pass

_generic_ops  = (Dot, Cross,
                 Grad, Curl, Rot, Div)
# ...

# ...
def gelatize(expr, dim):
    # ... in the case of a Lambda expression
    args = None
    if isinstance(expr, Lambda):
        args = expr.variables
        expr = expr.expr
    # ...

    # ... we first need to find the ordered list of generic operators
    ops = [a for a in preorder_traversal(expr) if isinstance(a, _generic_ops)]
    # ...

    # ...
    for i in ops:
        # if i = Grad(u) then type(i) is Grad
        op = type(i)

        new  = eval('{0}_{1}d'.format(op, dim))
        expr = expr.subs(op, new)
    # ...

    if args:
        return Lambda(args, expr)
    else:
        return expr
# ...



# ...
def test_0():
    x,y, a = symbols('x y a')

    # ...
    expr = x+y
    print '> expr := {0}'.format(expr)

    expr = LinearOperator(expr)
    print '> final := {0}'.format(expr)
    print('')
    # ...

    # ...
    expr = 2*x+y
    print '> expr := {0}'.format(expr)

    expr = LinearOperator(expr)
    print '> final := {0}'.format(expr)
    print('')
    # ...

    # ...
    expr = a*x+y
    print '> expr := {0}'.format(expr)

    expr = LinearOperator(expr)
    print '> final := {0}'.format(expr)
    # ...

    # ...
    expr = 2*a*x+y
    print '> expr := {0}'.format(expr)

    expr = LinearOperator(expr)
    print '> final := {0}'.format(expr)
    # ...
# ...

# ...
def test_1():
    u, v, a = symbols('u v a')

    # ...
    expr = u+v
    print '> expr := {0}'.format(expr)

    expr = dx(expr)
    print '> final := {0}'.format(expr)
    print('')
    # ...

    # ...
    expr = 2*u*v
    print '> expr := {0}'.format(expr)

    expr = dx(expr)
    print '> final := {0}'.format(expr)
    print('')
    # ...

    # ... dx should not operate on u^2,
    #     since we consider only linearized weak formulations
    expr = u*u
    print '> expr := {0}'.format(expr)

    expr = dx(expr)
    print '> final := {0}'.format(expr)
    print('')
    # ...
# ...

# ...
def test_2d_1():
    x,y = symbols('x y')

    u = Symbol('u')
    v = Symbol('v')

    a = Lambda((x,y,v,u), Dot(Grad(u), Grad(v)))
    print '> input := {0}'.format(a)

    expr = gelatize(a, dim=2)

    print '> final := {0}'.format(expr)
    print('')
# ...

# ...
def test_2d_2():
    x,y = symbols('x y')

    u = IndexedBase('u')
    v = IndexedBase('v')

    a = Lambda((x,y,v,u), Rot(u) * Rot(v) + Div(u) * Div(v) + 0.2 * Dot(u, v))
    print '> input := {0}'.format(a)

    expr = gelatize(a, dim=2)

    print '> final := {0}'.format(expr.expand())
    print('')
# ...

# ...
def test_2d_3():
    x,y = symbols('x y')

    u = Symbol('u')
    v = Symbol('v')

    a = Lambda((x,y,v,u), Cross(Curl(u), Curl(v)) + 0.2 * u * v)
    print '> input := {0}'.format(a)

    expr = gelatize(a, dim=2)

    print '> final := {0}'.format(expr.expand())
    print('')
# ...

# ...
def test_3d_1():
    x,y,z = symbols('x y z')

    u = Symbol('u')
    v = Symbol('v')

    a = Lambda((x,y,z,v,u), Dot(Grad(u), Grad(v)))
    print '> input := {0}'.format(a)

    expr = gelatize(a, dim=3)

    print '> final := {0}'.format(expr)
    print('')
# ...

# ...
def test_3d_2():
    x,y = symbols('x y')

    u = IndexedBase('u')
    v = IndexedBase('v')


    a = Lambda((x,y,v,u), Div(u) * Div(v) + 0.2 * Dot(u, v))
    print '> input := {0}'.format(a)

    expr = gelatize(a, dim=3)

    print '> final := {0}'.format(expr.expand())
    print('')
# ...

# ...
def test_3d_3():
    x,y = symbols('x y')

    u = IndexedBase('u')
    v = IndexedBase('v')

    a = Lambda((x,y,v,u), Dot(Curl(u), Curl(v)) + 0.2 * Dot(u, v))
    print '> input := {0}'.format(a)

    expr = gelatize(a, dim=3)

    print '> final := {0}'.format(expr.expand())
    print('')
# ...

# ...
def test_3d_4a():
    x,y = symbols('x y')

    u = IndexedBase('u')
    v = IndexedBase('v')

    b = Tuple(1.0, 0., 0.)

    a = Lambda((x,y,v,u), Dot(Curl(Cross(b,u)), Curl(Cross(b,v))) + 0.2 * Dot(u, v))
    print '> input := {0}'.format(a)

    expr = gelatize(a, dim=3)

    print '> final := {0}'.format(expr.expand())
    print('')
# ...

# ...
def test_3d_4b():
    """Alfven operator."""
    x,y = symbols('x y')

    u = IndexedBase('u')
    v = IndexedBase('v')

    b = IndexedBase('b')

    c0,c1,c2 = symbols('c0 c1 c2')

    a = Lambda((x,y,v,u), (  c0 * Dot(u, v)
                           - c1 * Div(u) * Div(v) + c2 *
                           Dot(Curl(Cross(b,u)), Curl(Cross(b,v)))))
    print '> input := {0}'.format(a)

    expr = gelatize(a, dim=3)

    print '> final := {0}'.format(expr.expand())
    print('')
# ...

if __name__ == '__main__':
    test_0()
    test_1()

    test_2d_1()
    test_2d_2()
    test_2d_3()

    test_3d_1()
    test_3d_2()
    test_3d_3()
    test_3d_4a()
    test_3d_4b()
