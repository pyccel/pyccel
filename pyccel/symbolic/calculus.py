# coding: utf-8

# TODO: - raise error if the weak form is not linear/bilinear on the test/trial

from pyccel.symbolic.gelato import glt_symbol

import numpy as np

from sympy.core.sympify import sympify
from sympy.simplify.simplify import simplify
from sympy import Symbol
from sympy import Lambda
from sympy import Function
from sympy import bspline_basis
from sympy import lambdify
from sympy import cos
from sympy import sin
from sympy import Rational
from sympy import diff
from sympy import Matrix
from sympy import latex
from sympy import Integral
from sympy import I as sympy_I
from sympy.core import Basic
from sympy.core.singleton import S
from sympy.simplify.simplify import nsimplify
from sympy.utilities.lambdify import implemented_function
from sympy.matrices.dense import MutableDenseMatrix
from sympy import Mul, Add
from sympy import Tuple
from sympy import postorder_traversal
from sympy import preorder_traversal

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

from pyccel.ast.core import IndexedVariable, IndexedElement, Variable


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
            return IndexedElement(self, *indices, **kw_args)
        else:
            return IndexedElement(self, indices, **kw_args)

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
    coordinate = None

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
    coordinate = 'x'
    pass

class dy(DifferentialOperator):
    coordinate = 'y'
    pass

class dz(DifferentialOperator):
    coordinate = 'z'
    pass

_partial_derivatives = (dx, dy, dz)
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
            return IndexedElement(self, *indices, **kw_args)
        else:
            return IndexedElement(self, indices, **kw_args)

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
            return IndexedElement(self, *indices, **kw_args)
        else:
            return IndexedElement(self, indices, **kw_args)

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
            return IndexedElement(self, *indices, **kw_args)
        else:
            return IndexedElement(self, indices, **kw_args)

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
            return IndexedElement(self, *indices, **kw_args)
        else:
            return IndexedElement(self, indices, **kw_args)

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
            return IndexedElement(self, *indices, **kw_args)
        else:
            return IndexedElement(self, indices, **kw_args)

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
_coord_registery = ['x', 'y', 'z']
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
            return IndexedElement(self, *indices, **kw_args)
        else:
            return IndexedElement(self, indices, **kw_args)

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
def dict_to_matrix(d, instructions=None, **settings):
    """
    converts a dictionary of expressions to a matrix

    d: dict
        dictionary of expressions

    instructions: list
        a list to keep track of the applied instructions.

    settings: dict
        dictionary for different settings
    """
    # ...
    assert(type(d) == dict)
    # ...

    # ...
    n_rows = 1
    n_cols = 1
    for key, values in list(d.items()):
        if key[0]+1 > n_rows:
            n_rows = key[0] + 1
        if key[1]+1 > n_cols:
            n_cols = key[1] + 1
    # ...

    # ...
    expressions = []
    for i_row in range(0, n_rows):
        row_expr = []
        for i_col in range(0, n_cols):
            _expr = None
            try:
                _expr = d[i_row,i_col]
            except:
                _expr = S.Zero
            row_expr.append(_expr)
        expressions.append(row_expr)
    # ...

    # ...
    expr = Matrix(expressions)
    # ...

    # ... updates the latex expression
    if instructions is not None:
        # ...
        title  = "GLT symbol"
        instructions.append(latex_title_as_paragraph(title))
        # ...

        # ...
        sets = {}
        for key, value in list(settings.items()):
            if not(key == "glt_integrate"):
                sets[key] = value

        instructions.append(glt_latex(expr, **sets))
        # ...
    # ...

    return expr
# ...

# ...
def initialize_weak_form(f, dim):
    if not isinstance(f, Lambda):
        raise TypeError('Expecting a Lambda')

    args = f.variables
    n_args = len(args)
    if (n_args - dim) % 2 == 1:
        raise ValueError('Wrong number of arguments')

    n = (n_args - dim) / 2

    coords = Tuple(*args[:dim])
    tests  = Tuple(*args[dim:dim+n])
    trials = Tuple(*args[dim+n:])

#    print('> coords : {0}'.format(coords))
#    print('> tests  : {0}'.format(tests))
#    print('> trials : {0}'.format(trials))

    test_names  = [str(i) for i in tests]
    trial_names = [str(i) for i in trials]
    coord_names = [str(i) for i in coords]

    d = {}
    d_args = {}
    # TODO must fix the precision for S.Zero?
    for i_test in range(0, n):
        for i_trial in range(0, n):
            d[(i_test, i_trial)] = S.Zero
            d_args[(i_test, i_trial)] = []


    # ...
    def _find_atom(expr, atom):
        """."""
        if not(isinstance(atom, (Symbol, IndexedVariable, Variable))):
            raise TypeError('Wrong type, given {0}'.format(type(atom)))

        if isinstance(expr, (list, tuple, Tuple)):
            ls = [_find_atom(i, atom) for i in expr]
            return np.array(ls).any()

        if isinstance(expr, Add):
            return _find_atom(expr._args, atom)

        if isinstance(expr, Mul):
            return _find_atom(expr._args, atom)

        if isinstance(expr, Function):
            return _find_atom(expr.args, atom)

        if isinstance(expr, IndexedElement):
            return (str(expr.base) == str(atom))

        if isinstance(expr, Variable):
            return (str(expr) == str(atom))

        if isinstance(expr, Symbol):
            return (str(expr) == str(atom))

        return False
    # ...

    # ...
    def _is_vector(expr, atom):
        """."""
        if not(isinstance(atom, (Symbol, IndexedVariable, Variable))):
            raise TypeError('Wrong type, given {0}'.format(type(atom)))

        if isinstance(expr, (list, tuple, Tuple)):
            ls = [_is_vector(i, atom) for i in expr]
            return np.array(ls).any()

        if isinstance(expr, Add):
            return _is_vector(expr._args, atom)

        if isinstance(expr, Mul):
            return _is_vector(expr._args, atom)

        if isinstance(expr, Function):
            return _is_vector(expr.args, atom)

        if isinstance(expr, IndexedElement):
            return True

        return False
    # ...

    # ... be careful here, we are using side effect on (d, d_args)
    def _decompose(expr):
        if isinstance(expr, Mul):
            for i_test, test in enumerate(tests):
                for i_trial, trial in enumerate(trials):
                    if _find_atom(expr, test) and _find_atom(expr, trial):
                        d[(i_test, i_trial)] += expr
                        d_args[(i_test, i_trial)] = Tuple(test, trial)
        elif isinstance(expr, Add):
            for e in expr._args:
                _decompose(e)
#        else:
#            raise NotImplementedError('given type {0}'.format(type(expr)))

        return d, d_args
    # ...

    expr = f.expr
    expr = expr.expand()
#    expr = expr.subs({Function('Grad'): Grad})
#    expr = expr.subs({Function('Dot'): Dot})

    d, d_args = _decompose(expr)

    d_expr = {}
    for k,expr in d.items():
        args = list(coords)

        found_vector = False
        for u in d_args[k]:
            if _is_vector(expr, u):
                found_vector = True
                for i in range(0, dim):
                    uofi = IndexedVariable(str(u))[i]
                    ui = Symbol('{0}{1}'.format(u, i+1))
                    expr = expr.subs(uofi, ui)
                    args += [ui]
            else:
                args += [u]

        d_expr[k] = Lambda(args, expr)
        if found_vector:
            d_expr[k], _infos = initialize_weak_form(d_expr[k], dim)

    if len(d_expr) == 1:
        key = d_expr.keys()[0]
        d_expr = d_expr[key]

    info = {}
    info['coords'] = coords
    info['tests']  = tests
    info['trials'] = trials

    return d_expr, info

# ...
def normalize_weak_from(f):
    """
    Converts an expression using dx, dy, etc to a normal form, where we
    introduce symbols with suffix to define derivatives.

    f: dict, Lambda
        a valid weak formulation in terms of dx, dy etc
    """
    # ...
    if type(f) == dict:
        d_expr = {}
        for key, g in list(f.items()):
            # ...
            d_expr[key] = normalize_weak_from(g)
            # ...

        return dict_to_matrix(d_expr)
    # ...

    # ...
    if not isinstance(f, Lambda):
        raise TypeError('Expecting a Lambda expression')
    # ...

    # ...
    expr = f.expr

    args   = f.variables
    n_args = len(args)
    # ...

    # ...
    coords = [i for i in f.variables if str(i) in _coord_registery]
    dim    = len(coords)

    if (n_args - dim) % 2 == 1:
        raise ValueError('Wrong number of arguments')

    n = (n_args - dim) / 2

    coords = Tuple(*args[:dim])
    tests  = Tuple(*args[dim:dim+n])
    trials = Tuple(*args[dim+n:])

    # ... we first need to find the ordered list of generic operators
    ops = [a for a in preorder_traversal(expr) if isinstance(a, _partial_derivatives)]
    # ...

    # ...
    for i in ops:
        # if i = dx(u) then type(i) is dx
        op = type(i)
        coordinate = op.coordinate
        for a in i.args:
            # ... test functions
            if a in tests:
                expr = expr.subs({i: Symbol('Ni_{0}'.format(coordinate))})

            if isinstance(a, IndexedElement) and a.base in tests:
                expr = expr.subs({i: Symbol('Ni_{0}'.format(coordinate))})
            # ...

            # ... trial functions
            if a in trials:
                expr = expr.subs({i: Symbol('Nj_{0}'.format(coordinate))})

            if isinstance(a, IndexedElement) and a.base in trials:
                expr = expr.subs({i: Symbol('Nj_{0}'.format(coordinate))})
            # ...
    # ...

    # ...
    for i in tests:
        expr = expr.subs({i: Symbol('Ni')})

    for i in trials:
        expr = expr.subs({i: Symbol('Nj')})
    # ...

    return expr
# ...


# ...
def test_0():
    x,y, a = symbols('x y a')

    # ...
    expr = x+y
    print '> expr := {0}'.format(expr)

    expr = LinearOperator(expr)
    print '> gelatized := {0}'.format(expr)
    print('')
    # ...

    # ...
    expr = 2*x+y
    print '> expr := {0}'.format(expr)

    expr = LinearOperator(expr)
    print '> gelatized := {0}'.format(expr)
    print('')
    # ...

    # ...
    expr = a*x+y
    print '> expr := {0}'.format(expr)

    expr = LinearOperator(expr)
    print '> gelatized := {0}'.format(expr)
    # ...

    # ...
    expr = 2*a*x+y
    print '> expr := {0}'.format(expr)

    expr = LinearOperator(expr)
    print '> gelatized := {0}'.format(expr)
    # ...
# ...

# ...
def test_1():
    u, v, a = symbols('u v a')

    # ...
    expr = u+v
    print '> expr := {0}'.format(expr)

    expr = dx(expr)
    print '> gelatized := {0}'.format(expr)
    print('')
    # ...

    # ...
    expr = 2*u*v
    print '> expr := {0}'.format(expr)

    expr = dx(expr)
    print '> gelatized := {0}'.format(expr)
    print('')
    # ...

    # ... dx should not operate on u^2,
    #     since we consider only linearized weak formulations
    expr = u*u
    print '> expr := {0}'.format(expr)

    expr = dx(expr)
    print '> gelatized := {0}'.format(expr)
    print('')
    # ...
# ...

# ...
def test_2d_1():
    x,y = symbols('x y')

    u = Symbol('u')
    v = Symbol('v')

    a = Lambda((x,y,v,u), Dot(Grad(u), Grad(v)) + u*v)
    print '> input       := {0}'.format(a)

    expr = gelatize(a, dim=2)
    print '> gelatized   := {0}'.format(expr)

    expr = normalize_weak_from(expr)
    print '> normal form := {0}'.format(expr)

    # ... create a glt symbol from a string without evaluation
    #     a discretization is defined as a dictionary
    discretization = {"n_elements": [16, 16], "degrees": [3, 3]}

    expr = glt_symbol(expr, dim=2, discretization=discretization, evaluate=False)
    print '> glt symbol  := {0}'.format(expr)
    # ...

    print('')
# ...

# ...
def test_2d_2():
    x,y = symbols('x y')

    u = IndexedVariable('u')
    v = IndexedVariable('v')

    a = Lambda((x,y,v,u), Rot(u) * Rot(v) + Div(u) * Div(v) + 0.2 * Dot(u, v))
    print '> input       := {0}'.format(a)

    expr = gelatize(a, dim=2)
    print '> gelatized   := {0}'.format(expr)

    expr, info = initialize_weak_form(expr, dim=2)
    print '> temp form   :='
    # for a nice printing, we print the dictionary entries one by one
    for key, value in expr.items():
        print '\t\t', key, '\t', value

    expr = normalize_weak_from(expr)
    print '> normal form := {0}'.format(expr)

    # ... create a glt symbol from a string without evaluation
    #     a discretization is defined as a dictionary
    discretization = {"n_elements": [16, 16], "degrees": [3, 3]}

    expr = glt_symbol(expr, dim=2, discretization=discretization, evaluate=False)
    print '> glt symbol  := {0}'.format(expr)
    # ...

    print('')
# ...

# ...
def test_2d_3():
    x,y = symbols('x y')

    u = Symbol('u')
    v = Symbol('v')

    a = Lambda((x,y,v,u), Cross(Curl(u), Curl(v)) + 0.2 * u * v)
    print '> input       := {0}'.format(a)

    expr = gelatize(a, dim=2)
    print '> gelatized   := {0}'.format(expr)

    expr, info = initialize_weak_form(expr, dim=2)
    print '> temp form   := {0}'.format(expr)

    expr = normalize_weak_from(expr)
    print '> normal form := {0}'.format(expr)

    # ... create a glt symbol from a string without evaluation
    #     a discretization is defined as a dictionary
    discretization = {"n_elements": [16, 16], "degrees": [3, 3]}

    expr = glt_symbol(expr, dim=2, discretization=discretization, evaluate=False)
    print '> glt symbol  := {0}'.format(expr)
    # ...

    print('')
# ...

# ...
def test_3d_1():
    x,y,z = symbols('x y z')

    u = Symbol('u')
    v = Symbol('v')

    a = Lambda((x,y,z,v,u), Dot(Grad(u), Grad(v)))
    print '> input       := {0}'.format(a)

    expr = gelatize(a, dim=3)
    print '> gelatized   := {0}'.format(expr)

    expr, info = initialize_weak_form(expr, dim=3)
    print '> temp form   := {0}'.format(expr)

    expr = normalize_weak_from(expr)
    print '> normal form := {0}'.format(expr)

    # ... create a glt symbol from a string without evaluation
    #     a discretization is defined as a dictionary
    discretization = {"n_elements": [16, 16, 16], "degrees": [3, 3, 3]}

    expr = glt_symbol(expr, dim=3, discretization=discretization, evaluate=False)
    print '> glt symbol  := {0}'.format(expr)
    # ...

    print('')
# ...

# ...
def test_3d_2():
    x,y,z = symbols('x y z')

    u = IndexedVariable('u')
    v = IndexedVariable('v')


    a = Lambda((x,y,z,v,u), Div(u) * Div(v) + 0.2 * Dot(u, v))
    print '> input       := {0}'.format(a)

    expr = gelatize(a, dim=3)
    print '> gelatized   := {0}'.format(expr)

    expr, info = initialize_weak_form(expr, dim=3)
    print '> temp form   :='
    # for a nice printing, we print the dictionary entries one by one
    for key, value in expr.items():
        print '\t\t', key, '\t', value

    expr = normalize_weak_from(expr)
    print '> normal form := {0}'.format(expr)

    # ... create a glt symbol from a string without evaluation
    #     a discretization is defined as a dictionary
    discretization = {"n_elements": [16, 16, 16], "degrees": [3, 3, 3]}

    expr = glt_symbol(expr, dim=3, discretization=discretization, evaluate=False)
    print '> glt symbol  := {0}'.format(expr)
    # ...

    print('')
# ...

# ...
def test_3d_3():
    x,y,z = symbols('x y z')

    u = IndexedVariable('u')
    v = IndexedVariable('v')

    a = Lambda((x,y,z,v,u), Dot(Curl(u), Curl(v)) + 0.2 * Dot(u, v))
    print '> input       := {0}'.format(a)

    expr = gelatize(a, dim=3)
    print '> gelatized   := {0}'.format(expr)

    expr, info = initialize_weak_form(expr, dim=3)
    print '> temp form   :='
    # for a nice printing, we print the dictionary entries one by one
    for key, value in expr.items():
        print '\t\t', key, '\t', value

    expr = normalize_weak_from(expr)
    print '> normal form := {0}'.format(expr)

    # ... create a glt symbol from a string without evaluation
    #     a discretization is defined as a dictionary
    discretization = {"n_elements": [16, 16, 16], "degrees": [3, 3, 3]}

    expr = glt_symbol(expr, dim=3, discretization=discretization, evaluate=False)
    print '> glt symbol  := {0}'.format(expr)
    # ...

    print('')
# ...

# ...
def test_3d_4a():
    x,y,z = symbols('x y z')

    u = IndexedVariable('u')
    v = IndexedVariable('v')

    b = Tuple(1.0, 0., 0.)

    a = Lambda((x,y,z,v,u), Dot(Curl(Cross(b,u)), Curl(Cross(b,v))) + 0.2 * Dot(u, v))
    print '> input       := {0}'.format(a)

    expr = gelatize(a, dim=3)
    print '> gelatized   := {0}'.format(expr)

    expr, info = initialize_weak_form(expr, dim=3)
    print '> temp form   :='
    # for a nice printing, we print the dictionary entries one by one
    for key, value in expr.items():
        print '\t\t', key, '\t', value

    expr = normalize_weak_from(expr)
    print '> normal form := {0}'.format(expr)

    # ... create a glt symbol from a string without evaluation
    #     a discretization is defined as a dictionary
    discretization = {"n_elements": [16, 16, 16], "degrees": [3, 3, 3]}

    expr = glt_symbol(expr, dim=3, discretization=discretization, evaluate=False)
    print '> glt symbol  := {0}'.format(expr)
    # ...

    print('')
# ...

# ...
def test_3d_4b():
    """Alfven operator."""
    x,y,z = symbols('x y z')

    u = IndexedVariable('u')
    v = IndexedVariable('v')

    b = IndexedVariable('b')

    c0,c1,c2 = symbols('c0 c1 c2')

    a = Lambda((x,y,z,v,u), (  c0 * Dot(u, v)
                             - c1 * Div(u) * Div(v)
                             + c2 *Dot(Curl(Cross(b,u)), Curl(Cross(b,v)))))
    print '> input       := {0}'.format(a)

    expr = gelatize(a, dim=3)
    print '> gelatized   := {0}'.format(expr)

    # TODO: fix, not working
#    expr, info = initialize_weak_form(expr, dim=3)
#    print '> temp form   :='
#    # for a nice printing, we print the dictionary entries one by one
#    for key, value in expr.items():
#        print '\t\t', key, '\t', value
#
#    expr = normalize_weak_from(expr)
#    print '> normal form := {0}'.format(expr)
#
#    # ... create a glt symbol from a string without evaluation
#    #     a discretization is defined as a dictionary
#    discretization = {"n_elements": [16, 16, 16], "degrees": [3, 3, 3]}
#
#    expr = glt_symbol(expr, dim=3, discretization=discretization, evaluate=False)
#    print '> glt symbol  := {0}'.format(expr)
#    # ...

    print('')
# ...

if __name__ == '__main__':
#    test_0()
#    test_1()

#    test_2d_1()
#    test_2d_2()
#    test_2d_3()

    test_3d_1()
    test_3d_2()
    test_3d_3()
    test_3d_4a()
    test_3d_4b()
