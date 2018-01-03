# -*- coding: utf-8 -*-
#
#
# TODO use to_assign and post processing as expression and not latex => helpful
#      for Fortran and Lua (code gen).
"""This module contains different functions to create and treate the GLT symbols."""

__all__ = ["glt_formatting", "glt_formatting_atoms"]

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
from sympy import Mul
from sympy import Tuple
from sympy import postorder_traversal
from sympy import preorder_traversal

from itertools import product

import numpy as np

# TODO find a better solution.
#      this code is duplicated in printing.latex
ARGS_x       = ["x", "y", "z"]
ARGS_u       = ["u", "v", "w"]
ARGS_s       = ["s", "ss"]
BASIS_TEST   = "Ni"
BASIS_TRIAL  = "Nj"
BASIS_PREFIX = ["x", "y", "z", "xx", "yy", "zz", "xy", "yz", "xz"]
TOLERANCE    = 1.e-10
#TOLERANCE    = 1.e-4
SETTINGS     = ["glt_integrate", "glt_formatting", "glt_formatting_atoms"]


# ...
_coord_registery = ['x', 'y', 'z']
_basis_registery = ['Ni',
                    'Ni_x', 'Ni_y', 'Ni_z',
                    'Ni_xx', 'Ni_yy', 'Ni_zz',
                    'Ni_xy', 'Ni_yz', 'Ni_zx',
                    'Nj',
                    'Nj_x', 'Nj_y', 'Nj_z',
                    'Ni_xx', 'Ni_yy', 'Ni_zz',
                    'Ni_xy', 'Ni_yz', 'Ni_zx']

dx = Function('dx')
dy = Function('dy')
dz = Function('dz')

dxx = Function('dxx')
dyy = Function('dyy')
dzz = Function('dzz')
dxy = Function('dxy')
dyz = Function('dyz')
dzx = Function('dzx')

# TODO how to treat 1d, 2d, 3d etc?
grad = lambda u: (dx(u), dy(u))
curl = lambda u: dy(u[0]) - dx(u[1])
rot  = lambda u: (dy(u), -dx(u))
div  = lambda u: dx(u[0]) + dy(u[1])
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
    for i_test in range(0, n):
        for i_trial in range(0, n):
            d[(i_test, i_trial)] = S.Zero
            d_args[(i_test, i_trial)] = []

    expr = f.expr
    for arg in preorder_traversal(expr):
        if isinstance(arg, Mul):
            found_test  = False
            found_trial = False

            test  = None
            trial = None

            for a in preorder_traversal(arg):
                if isinstance(a, Function):
                    pass
                elif isinstance(a, Symbol):
                    if str(a) in test_names:
                        found_test  = True
                        test = a
                    if str(a) in trial_names:
                        found_trial  = True
                        trial = a
            i_test = tests.index(test)
            i_trial = trials.index(trial)

            d[(i_test, i_trial)] += arg
            d_args[(i_test, i_trial)] = Tuple(test, trial)

    expr = {}
    for k,e in d.items():
        args = list(coords) + list(d_args[k])
        expr[k] = Lambda(args, e)

    info = {}
    info['coords'] = coords
    info['tests']  = tests
    info['trials'] = trials

    return expr, info


# ... TODO works only for scalar cases
def normalize_weak_from(f):
    """
    Converts an expression using dx, dy, etc to a normal form, where we
    introduce symbols with suffix to define derivatives.
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

    args = [i for i in f.variables if str(i) not in _coord_registery]
#    expr = sympify(f.expr)
    expr = f.expr

    for d in ['dx', 'dy', 'dz']:
        for i, arg in enumerate(args):
            atom = sympify('{0}({1})'.format(d, arg))
            suffix = None
            if i == 0:
                suffix = 'i'
            elif i == 1:
                suffix = 'j'
            expr = expr.subs({atom: Symbol('N{0}_{1}'.format(suffix, d[1]))})

    for i, arg in enumerate(args):
        atom = sympify('{0}'.format(arg))
        suffix = None
        if i == 0:
            suffix = 'i'
        elif i == 1:
            suffix = 'j'
        expr = expr.subs({atom: Symbol('N{0}'.format(suffix))})

    return expr
# ...

# ...
class weak_formulation(Function):
    """

    Examples
    ========

    """

    nargs = None

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

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        # ...
        f = _args[0]
        # ...

        # TODO must be computed somehow
        dim = 2

        # ...
        f, info = initialize_weak_form(f, dim)

        coords = info['coords']
        tests  = info['tests']
        trials = info['trials']

        test_names  = [str(i) for i in tests]
        trial_names = [str(i) for i in trials]
        coord_names = [str(i) for i in coords]
        # ...

        # ...
        expr = normalize_weak_from(f)
        # ...

        # ... TODO improve
        free_symbols = [str(i) for i in expr.free_symbols]
        free_symbols.sort()

        args  = _coord_registery[:dim]
        args += [i for i in free_symbols if i in _basis_registery]

        args = [Symbol(i) for i in args]
        # ...

        expr = Lambda(args, expr)

        return expr
# ...

# ...
def basis_symbols(dim, n_deriv=1):
    """
    Returns a dictionary that contains sympy symbols for the basis and their
    derivatives. The kind of a basis function can be trial or test.

    dim: int
        dimension of the logical/physical domain.

    n_deriv: int
        number of derivatives
    """
    # ...
    args_x = ARGS_x[:dim]
    # ...

    # ...
    words = []
    for i_deriv in range(1, n_deriv+1):
        words += [''.join(i) for i in product(args_x, repeat = i_deriv)]
    # ...

    # ...
    ops = [o for o in words if o in BASIS_PREFIX]
    # ...

    # ...
    ns = {}

    for B in [BASIS_TEST, BASIS_TRIAL]:
        ns[B] = Symbol(B)
        for d in ops:
            B_d = B + "_" + d
            ns[B_d] = Symbol(B_d)
    # ...

    return ns
# ...

# ...
def apply_mapping(expr, dim, instructions=None, **settings):
    """
    Applies a mapping to a given expression

    expr: sympy.Expression
        a sympy expression

    dim: int
        dimension of the logical/physical domain.

    instructions: list
        a list to keep track of the applied instructions.

    settings: dict
        dictionary for different settings
    """
    # ...
    args_x = ARGS_x[:dim]
    args_u = ARGS_u[:dim]
    for B in [BASIS_TEST, BASIS_TRIAL]:
        for (x,u) in zip(args_x, args_u):
            B_x = B + "_" + x
            B_u = B + "_" + u
            expr = expr.subs({Symbol(B_x): Symbol(B_u)})
    # ...

    # ... updates the latex expression
    if instructions is not None:
        sets = {}
        for key, value in list(settings.items()):
            sets[key] = value
        sets["mode"] = "equation*"

        instructions.append(glt_latex(expr, **sets))
    # ...

    return expr
# ...

# ...
def apply_tensor(expr, dim, instructions=None, **settings):
    """
    decomposes the basis function to their tensor form

    expr:
        a sympy expression

    dim: int
        dimension of the logical/physical domain.

    instructions: list
        a list to keep track of the applied instructions.

    settings: dict
        dictionary for different settings
    """
    args_u = ARGS_u[:dim]
    for B in [BASIS_TEST, BASIS_TRIAL]:
        # ... substruct the basis function
        prod = S.One
        for k in range(0, dim):
            Bk_u = B + str(k+1)
            prod *= Symbol(Bk_u)
        expr = expr.subs({Symbol(B): prod})
        # ...

        # ... substruct the derivatives
        for i,u in enumerate(args_u):
            B_u = B + "_" + u
            prod = S.One
            for k in range(0, dim):
                if k==i:
                    Bk_u = B + str(k+1) + "_s"
                else:
                    Bk_u = B + str(k+1)
                prod *= Symbol(Bk_u)
            expr = expr.subs({Symbol(B_u): prod})
        # ...

    # ... updates the latex expression
    if instructions is not None:
        sets = {}
        for key, value in list(settings.items()):
            sets[key] = value
        sets["mode"] = "equation*"

        instructions.append(glt_latex(expr, **sets))
    # ...

    return expr
# ...

# ...
def apply_factor(expr, dim, instructions=None, **settings):
    """
    factorizes the basis function by coupling the trial/test functions related
    to the same tensor index.

    expr:
        a sympy expression

    dim: int
        dimension of the logical/physical domain.

    instructions: list
        a list to keep track of the applied instructions.

    settings: dict
        dictionary for different settings
    """
    Bi = BASIS_TEST
    Bj = BASIS_TRIAL

    for k in range(0, dim):
        # ... mass symbol
        Bik = Bi + str(k+1)
        Bjk = Bj + str(k+1)
        P = Symbol(Bik) * Symbol(Bjk)
        mk = Symbol("m"+str(k+1))

        expr = expr.subs({P: mk})
        # ...

        # ... stiffness symbol
        Bik = Bi + str(k+1) + "_s"
        Bjk = Bj + str(k+1) + "_s"
        P = Symbol(Bik) * Symbol(Bjk)
        sk = Symbol("s"+str(k+1))

        expr = expr.subs({P: sk})
        # ...

        # ... advection symbol
        Bik = Bi + str(k+1)
        Bjk = Bj + str(k+1) + "_s"
        P = Symbol(Bik) * Symbol(Bjk)
        ak = Symbol("a"+str(k+1))

        expr = expr.subs({P: ak})
        # ...

        # ... adjoint advection symbol
        Bik = Bi + str(k+1) + "_s"
        Bjk = Bj + str(k+1)
        P = Symbol(Bik) * Symbol(Bjk)
        ak = Symbol("a"+str(k+1))

        expr = expr.subs({P: -ak})
        # ...

    # ... updates the latex expression
    if instructions is not None:
        # ...
        instruction = "The symbol is then:"
        instructions.append(instruction)
        # ...

        # ...
        sets = {}
        for key, value in list(settings.items()):
            if not(key == "glt_integrate"):
                sets[key] = value
        sets["mode"] = "equation*"

        instructions.append(glt_latex(expr, **sets))
        # ...
    # ...

    return expr
# ...

# ...
def glt_update_atoms(expr, discretization):
    """
    updates the glt symbol with the atomic symbols

    expr:
        a sympy expression

    discretization: dict
        a dictionary that contains the used discretization
    """
    # ...
    dim = len(discretization["n_elements"])
    # ...

    # ...
    args = _coord_registery[:dim]
    args = [Symbol(i) for i in args]
    # ...

    # ...
    for k in range(0, dim):
        # ...
        t = Symbol('t'+str(k+1))

        n = discretization["n_elements"][k]
        p = discretization["degrees"][k]

        m   = glt_symbol_m(n,p,t)
        s   = glt_symbol_s(n,p,t)
        a   = glt_symbol_a(n,p,t)
        t_a = -a
        # ...

        # ...
        expr = expr.subs({Symbol('m'+str(k+1)): m})
        expr = expr.subs({Symbol('s'+str(k+1)): s})
        expr = expr.subs({Symbol('a'+str(k+1)): a})
        expr = expr.subs({Symbol('t_a'+str(k+1)): t_a})
        # ...

        # ...
        args += [t]
        # ...
    # ...

    return Lambda(args, expr)
# ...

# ...
def glt_update_user_functions(expr, user_functions):
    """
    updates the glt symbol with user defined functions

    expr:
        a sympy expression

    user_functions: dict
        a dictionary containing the user defined functions
    """
    from clapp.vale.expressions.function import Function as CLAPP_Function
    for f_name, f in list(user_functions.items()):
        # ...
        if type(f) == CLAPP_Function:
            sympy_f = f.to_sympy()
        else:
            sympy_f = implemented_function(Function(f_name), f)
        # ...

        # ...
        expr = expr.subs({Symbol(f_name): sympy_f})
        # ...

    return expr
# ...

# ...
def glt_update_user_constants(expr, user_constants):
    """
    updates the glt symbol with user defined constants

    expr:
        a sympy expression

    user_constants: dict
        a dictionary containing the user defined constants
    """
    for f_name, f in list(user_constants.items()):
        # ...
        if type(f) in [int, float, complex]:
            expr = expr.subs({Symbol(f_name): f})
        # ...

    return expr
# ...

# ...
def glt_symbol(expr, dim, n_deriv=1, \
               verbose=False, evaluate=True, \
               discretization=None, \
               user_functions=None, \
               user_constants=None, \
               instructions=[], \
               **settings):
    """
    computes the glt symbol of a weak formulation given as a sympy expression.

    expr: sympy.Expression
        a sympy expression or a text

    dim: int
        dimension of the logical/physical domain.

    n_deriv: int
        maximum derivatives that appear in the weak formulation.

    verbose: bool
        talk more

    evaluate: bool
        causes the evaluation of the atomic symbols, if true

    discretization: dict
        a dictionary that contains the used discretization

    user_functions: dict
        a dictionary containing the user defined functions

    user_constants: dict
        a dictionary containing the user defined constants

    instructions: list
        a list to keep track of the applied instructions.

    settings: dict
        dictionary for different settings

    """
    # ...
    if verbose:
        print(("*** Input expression : ", expr))
    # ...

    # ...
    if type(expr) == dict:
        d_expr = {}
        for key, txt in list(expr.items()):
            # ... when using vale, we may get also a coefficient.
            if type(txt) == list:
                txt = str(txt[0]) + " * (" + txt[1] + ")"
            # ...

            # ...
            title  = "Computing the GLT symbol for the block " + str(key)
            instructions.append(latex_title_as_paragraph(title))
            # ...

            # ...
            d_expr[key] = glt_symbol(txt, dim, \
                                     n_deriv=n_deriv, \
                                     verbose=verbose, \
                                     evaluate=evaluate, \
                                     discretization=discretization, \
                                     user_functions=user_functions, \
                                     user_constants=user_constants, \
                                     instructions=instructions, \
                                     **settings)
            # ...

        return dict_to_matrix(d_expr, instructions=instructions, **settings)
    else:
        # ...
        ns = {}
        # ...

        # ...
        if user_constants is not None:
            for c_name, c in list(user_constants.items()):
                ns[c_name] = Symbol(c_name)
        # ...

        # ...
        d = basis_symbols(dim,n_deriv)
        for key, item in list(d.items()):
            ns[key] = item
        # ...

        # ...
        if isinstance(expr, Lambda):
            expr = normalize_weak_from(expr)
        # ...

        # ...
        expr = sympify(str(expr), locals=ns)
        # ...

        # ... remove _0 for a nice printing
        #     TODO remove
        expr = expr.subs({Symbol("Ni_0"): Symbol("Ni")})
        expr = expr.subs({Symbol("Nj_0"): Symbol("Nj")})
        # ...
    # ...

    # ...
    if verbose:
        # ...
        instruction = "We consider the following weak formulation:"
        instructions.append(instruction)
        instructions.append(glt_latex(expr, **settings))
        # ...

        print((">>> weak formulation: ", expr))
    # ...

    # ...
    expr = apply_mapping(expr, dim=dim, \
                         instructions=instructions, \
                         **settings)
    if verbose:
        print(expr)
    # ...

    # ...
    expr = apply_tensor(expr, dim=dim, \
                         instructions=instructions, \
                         **settings)
    if verbose:
        print(expr)
    # ...

    # ...
    expr = apply_factor(expr, dim, \
                         instructions=instructions, \
                         **settings)
    if verbose:
        print(expr)
    # ...

    # ...
    if not evaluate:
        return expr
    # ...

    # ...
    if not discretization:
        return expr
    # ...

    # ...
    expr = glt_update_atoms(expr, discretization)
    # ...

    # ...
    if (not user_functions) and (not user_constants):
        return expr
    # ...

    # ...
    if user_constants:
        expr = glt_update_user_constants(expr, user_constants)
    # ...

    # ...
    if user_functions:
        expr = glt_update_user_functions(expr, user_functions)
    # ...

    return expr
# ...

# ...
def glt_symbol_from_weak_formulation(form, discretization, \
                                     user_constants=None, \
                                     verbose=False, evaluate=True, \
                                     instructions=[], \
                                     **settings):
    """
    creates a glt symbol from a weak formulation.

    form: vale.BilinearForm
        a weak formulation.

    discretization: dict
        a dictionary that contains the used discretization

    user_constants: dict
        a dictionary containing the user defined constants

    verbose: bool
        talk more

    evaluate: bool
        causes the evaluation of the atomic symbols, if true

    instructions: list
        a list to keep track of the applied instructions.

    settings: dict
        dictionary for different settings
    """
    # ... TODO sets n_deriv from bilinear form
    n_deriv = 2
    # ...

    # ... gets the dimension
    dim = form.assembler.trial_space.context.p_dim
    # ...

    # ... TODO user constants from form
    # we consider form to be sympy expression for the moment
    expr = glt_symbol(form.glt_expr, dim, n_deriv=n_deriv, \
                      verbose=verbose, evaluate=evaluate, \
                      discretization=discretization, \
                      user_functions=form.functions, \
                      user_constants=user_constants, \
                      instructions=instructions, \
                      **settings)
    # ...

    return expr
# ...

# ...
def glt_lambdify(expr, dim=None, discretization=None):
    """
    it is supposed that glt_symbol has been called before.

    expr: sympy.Expression
        a sympy expression or a text

    dim: int
        dimension of the logical/physical domain.

    discretization: dict
        a dictionary that contains the used discretization
    """
    _dim = dim
    if dim is None:
        if discretization is not None:
            _dim = len(discretization["n_elements"])
        else:
            raise ValueError("> either dim or discretization must be provided.")

    args_x = ["x","y","z"]
    args_t = ["t1","t2","t3"]
    args_xt = args_x[:_dim] + args_t[:_dim]
    args = [Symbol(x) for x in args_xt]
    return lambdify(args, expr, "numpy")
# ...

# ...
def glt_approximate_eigenvalues(expr, discretization, mapping=None):
    """
    approximates the eigenvalues using a uniform sampling

    expr: sympy.Expression
        a sympy expression or a text

    discretization: dict
        a dictionary that contains the used discretization

    mapping: clapp.spl.mapping.Mapping
        a mapping object (geometric transformation)
    """
    # ...
    is_block = False
    # ...

    # ... lambdify the symbol.
    #     The block case will be done later.
    if type(expr) == MutableDenseMatrix:
        is_block = True
    else:
        f = glt_lambdify(expr, discretization=discretization)
    # ...

    # ...
    n       = discretization['n_elements']
    degrees = discretization['degrees']

    dim     = len(n)
    # ...

    # ...
    if dim == 1:
        # TODO boundary condition
        nx = n[0] + degrees[0] - 2

        t1 = np.linspace(-np.pi,np.pi, nx)

        u = np.linspace(0.,1.,nx)
        if mapping is not None:
            x = mapping.evaluate(u)[0,:]
        else:
            x = u

        if is_block:
            eigen = expr.eigenvals()

            eigs = []
            for ek, mult in list(eigen.items()):
                f = glt_lambdify(ek, discretization=discretization)
                t = f(x,t1)
                eigs += mult * list(t)

            return np.asarray(eigs) + 0.j
        else:
            return f(x,t1)
    elif dim == 2:
        # TODO boundary condition
        nx = n[0] + degrees[0] - 2
        ny = n[1] + degrees[1] - 2

        t1 = np.linspace(-np.pi,np.pi, nx)
        t2 = np.linspace(-np.pi,np.pi, ny)

        u = np.linspace(0.,1.,nx)
        v = np.linspace(0.,1.,ny)
        if mapping is not None:
            x = mapping.evaluate(u,v)[0,:,:]
            y = mapping.evaluate(u,v)[1,:,:]
        else:
            x,y   = np.meshgrid(u,v)

        t1,t2 = np.meshgrid(t1,t2)

        if is_block:
            eigen = expr.eigenvals()

            eigs = []
            for ek, mult in list(eigen.items()):
                f = glt_lambdify(ek, discretization=discretization)
                t = f(x,y,t1,t2).ravel()
                eigs += mult * list(t)

            return np.asarray(eigs) + 0.j
        else:
            rr = f(x,y,t1,t2)
            return f(x,y,t1,t2).ravel()
    elif dim == 3:
        # TODO boundary condition
        nx = n[0] + degrees[0] - 2
        ny = n[1] + degrees[1] - 2
        nz = n[2] + degrees[2] - 2

        t1 = np.linspace(-np.pi,np.pi, nx)
        t2 = np.linspace(-np.pi,np.pi, ny)
        t3 = np.linspace(-np.pi,np.pi, nz)

        u = np.linspace(0.,1.,nx)
        v = np.linspace(0.,1.,ny)
        w = np.linspace(0.,1.,nz)
        if mapping is not None:
            x = mapping.evaluate(t1,t2,t3)[0,:,:,:]
            y = mapping.evaluate(t1,t2,t3)[1,:,:,:]
            z = mapping.evaluate(t1,t2,t3)[2,:,:,:]
        else:
            # 1 and 0  are inverted to get the right shape
            x,y,z = np.meshgrid(t2,t1,t3)

        # 1 and 0  are inverted to get the right shape
        t1,t2,t3 = np.meshgrid(t2,t1,t3)

        if is_block:
            eigen = expr.eigenvals()

            eigs = []
            for ek, mult in list(eigen.items()):
                f = glt_lambdify(ek, discretization=discretization)
                t = f(x,y,z,t1,t2,t3).ravel()
                eigs += mult * list(t)

            return np.asarray(eigs) + 0.j
        else:
            return f(x,y,z,t1,t2,t3).ravel()
    # ...
# ...

# ...
def glt_plot_eigenvalues(expr, discretization, \
                         mapping=None, \
                         matrix=None, \
                         tolerance=1.e-8, **settings):
    """
    plots the approximations of the eigenvalues by means of a uniform sampling
    of the glt symbol.

    expr: sympy.Expression
        a sympy expression or a text

    discretization: dict
        a dictionary that contains the used discretization

    mapping: clapp.spl.mapping.Mapping
        a mapping object (geometric transformation)

    matrix: clapp.plaf.matrix.Matrix
        a matrix object after assembling the weak formulation.

    tolerance: float
        a tolerance to check if the values are pure real numbers.

    settings: dict
        dictionary for different settings
    """
    import matplotlib.pyplot as plt

    # ...
    M = None
    if matrix is not None:
        from scipy.linalg import eig

        # ... PLAF matrix or scipy sparse
        from clapp.plaf.matrix import Matrix
        if type(matrix) == Matrix:
            M = matrix.get().todense()
        elif type(matrix) == dict:
            raise ValueError("NOT YET IMPLEMENTED")
        else:
            M = matrix.todense()
        # ...
    # ...

    # ...
    try:
        label = settings["label"]
    except:
        label = "glt symbol"
    # ...

    # ...
    try:
        properties = settings["properties"]
    except:
        properties = "+b"
    # ...

    # ... uniform sampling of the glt symbol
    t  = glt_approximate_eigenvalues(expr, discretization, mapping=mapping)

    tr = t.real.ravel()
    ti = t.imag.ravel()
    # ...

    # ... real case
    if (np.linalg.norm(ti) < tolerance):
        # ...
        tr.sort()

        plt.plot(tr, properties, label=label)
        # ...

        # ...
        if M is not None:
            # ...
            w, v = eig(M)
            wr = w.real
            wr.sort()
            plt.plot(wr, "xr", label="eigenvalues")
            # ...
        # ...
    else:
        # ...
        plt.plot(tr, ti, properties, label=label)
        # ...

        # ...
        if M is not None:
            # ...
            w, v = eig(M)
            wr = w.real
            wi = w.imag
            plt.plot(wr, wi, "xr", label="eigenvalues")
            # ...
        # ...
    # ...
# ...

# ...
class glt_function(Function):
    """

    Examples
    ========

    """

    nargs = None

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

    @classmethod
    def eval(cls, *_args):
        """."""

        if not _args:
            return

        f = _args[0]
        n = _args[1]
        p = _args[2]

        discretization = {"n_elements": n, "degrees": p}
        dim = len(n)

        f, info = initialize_weak_form(f, dim)

        coords = info['coords']
        tests  = info['tests']
        trials = info['trials']

        test_names  = [str(i) for i in tests]
        trial_names = [str(i) for i in trials]
        coord_names = [str(i) for i in coords]

        F = glt_symbol(f, dim=2, discretization=discretization, evaluate=True)

        # glt_symbol may return a matrix of lambdas
        if isinstance(F, Matrix):
            expressions = []
            for i in range(0, F.shape[0]):
                row = []
                for j in range(0, F.shape[1]):
                    row += [F[i,j].expr]
                expressions += [row]
            args = list(coords)
            args += [a for a in F[i,j].variables if not(str(a) in coord_names)]
            F = Lambda(args, Matrix(expressions))

        return F
# ...

# ...
class glt_symbol_m(Function):
    """
    A class for the mass symbol
    """
    nargs = 3

    @classmethod
    def eval(cls, n, p, t):
        # ...
        if not 0 <= p:
            raise ValueError("must have 0 <= p")
        if not 0 <= n:
            raise ValueError("must have 0 <= n")
        # ...

        # ...
        r  = Symbol('r')

        pp = 2*p + 1
        N = pp + 1
        L = list(range(0, N + pp + 1))

        b0 = bspline_basis(pp, L, 0, r)
        bsp = lambdify(r, b0)
        # ...

        # ... we use nsimplify to get the rational number
        phi = []
        for i in range(0, p+1):
            y = bsp(p+1-i)
            y = nsimplify(y, tolerance=TOLERANCE, rational=True)
            phi.append(y)
        # ...

        # ...
        m = phi[0] * cos(S.Zero)
        for i in range(1, p+1):
            m += 2 * phi[i] * cos(i * t)
        # ...

        # ... scaling
        m *= Rational(1,n)
        # ...

        return m
# ...

# ...
class glt_symbol_s(Function):
    """
    A class for the stiffness symbol
    """
    nargs = 3

    @classmethod
    def eval(cls, n, p, t):
        # ...
        if not 0 <= p:
            raise ValueError("must have 0 <= p")
        if not 0 <= n:
            raise ValueError("must have 0 <= n")
        # ...

        # ...
        r  = Symbol('r')

        pp = 2*p + 1
        N = pp + 1
        L = list(range(0, N + pp + 1))

        b0    = bspline_basis(pp, L, 0, r)
        b0_r  = diff(b0, r)
        b0_rr = diff(b0_r, r)
        bsp   = lambdify(r, b0_rr)
        # ...

        # ... we use nsimplify to get the rational number
        phi = []
        for i in range(0, p+1):
            y = bsp(p+1-i)
            y = nsimplify(y, tolerance=TOLERANCE, rational=True)
            phi.append(y)
        # ...

        # ...
        m = -phi[0] * cos(S.Zero)
        for i in range(1, p+1):
            m += -2 * phi[i] * cos(i * t)
        # ...

        # ... scaling
        m *= n
        # ...

        return m
# ...

# ...
class glt_symbol_a(Function):
    """
    A class for the advection symbol
    """
    nargs = 3

    @classmethod
    def eval(cls, n, p, t):
        # ...
        if not 0 <= p:
            raise ValueError("must have 0 <= p")
        if not 0 <= n:
            raise ValueError("must have 0 <= n")
        # ...

        # ...
        r  = Symbol('r')

        pp = 2*p + 1
        N = pp + 1
        L = list(range(0, N + pp + 1))

        b0   = bspline_basis(pp, L, 0, r)
        b0_r = diff(b0, r)
        bsp  = lambdify(r, b0_r)
        # ...

        # ... we use nsimplify to get the rational number
        phi = []
        for i in range(0, p+1):
            y = bsp(p+1-i)
            y = nsimplify(y, tolerance=TOLERANCE, rational=True)
            phi.append(y)
        # ...

        # ...
        m = -phi[0] * cos(S.Zero)
        for i in range(1, p+1):
            m += -2 * phi[i] * sin(i * t)
        # ...

        # ... make it pure imaginary
        m *= sympy_I
        # ...

        return m
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
def glt_symbol_laplace(discretization, \
                       verbose=False, evaluate=True, \
                       instructions=[], \
                       **settings):
    """
    Returns the Laplace symbol for a given discretization.

    discretization: dict
        a dictionary that contains the used discretization

    verbose: bool
        talk more

    evaluate: bool
        causes the evaluation of the atomic symbols, if true

    instructions: list
        a list to keep track of the applied instructions.

    settings: dict
        dictionary for different settings
    """
    # ...
    dim = len(discretization["n_elements"])
    # ...

    # ...
    if dim == 1:
        txt = "Ni_x * Nj_x"
    elif dim == 2:
        txt = "Ni_x * Nj_x + Ni_y * Nj_y"
    elif dim == 3:
        txt = "Ni_x * Nj_x + Ni_y * Nj_y + Ni_z * Nj_z"
    # ...

    # ...
    expr = glt_symbol(txt, dim, \
                      verbose=verbose, evaluate=evaluate, \
                      discretization=discretization, \
                      instructions=instructions, \
                      **settings)
    # ...

    return expr
# ...

# ...
def glt_integrate(expr, domain="Omega"):
    """
    Adds the integral to the expression. needed for printing.

    domain: str
        domain over which we integrate the expression.
    """
    return Integral(expr, domain)
# ...

# ...
def glt_formatting(expr, **settings):
    """
    Formatting the glt symbol, prior to calling a printer

    expr: sympy.Expression
        a sympy expression

    settings: dict
        dictionary for different settings
    """

    # ...
    try:
        domain = settings["glt_integrate"]
        domain = sympify(str(domain))

        expr = glt_integrate(expr, domain)
    except:
        pass
    # ...

    # ...
    try:
        fmt = settings["glt_formatting_atoms"]
        if fmt:
            expr = glt_formatting_atoms(expr, **settings)
    except:
        pass
    # ...

    return expr
# ...

# ...
def glt_formatting_atoms(expr, **settings):
    """
    Formatting the glt symbol atoms, prior to calling a printer

    expr: sympy.Expression
        a sympy expression

    settings: dict
        dictionary for different settings
    """
    # TODO do we still need it?

    # ...
    dim    = 3
    prefix = "\mathfrak{"
    suffix = "}"
    # ...

    # ...
    for k in range(0, dim):
        # ...
        t = Symbol('t'+str(k+1))

        for s in ["m", "s", "a"]:
            sk = s + str(k+1)
            s_old = Symbol(sk)
            s_new = Symbol(prefix + sk + suffix)

            expr = expr.subs({s_old: s_new})
        # ...
    # ...

    return expr
# ...



# ...
def latex_title_as_paragraph(title):
    """
    Returns the title as a paragraph.

    title: str
        a string for the paragraph title
    """
    return "\paragraph{" + str(title) + "}"
# ...

# ...
def glt_latex_definitions():
    """
    Returns the definitions of the atomic symbols for the GLT.
    """
    # ...
    t = Symbol('t')
    m = Symbol('m')
    s = Symbol('s')
    a = Symbol('a')
    # ...

    # ...
    def formula(symbol):
        """
        returns the latex formula for the mass symbol.
        """
        txt_m = r"\phi_{2p+1}(p+1) + 2 \sum_{k=1}^p \phi_{2p+1}(p+1-k) \cos(k \theta)"
        txt_s = r"- {\phi}''_{2p+1}(p+1) - 2 \sum_{k=1}^p {\phi}''_{2p+1}(p+1-k) \cos(k \theta)"
        txt_a = r"\phi_{2p+1}(p+1) + 2 \sum_{k=1}^p \phi_{2p+1}(p+1-k) \cos(k \theta)"

        if str(symbol) == "m":
            return txt_m
        elif str(symbol) == "s":
            return txt_s
        elif str(symbol) == "a":
            return txt_a
        else:
            print ("not yet available.")
    # ...

    # ...
    definitions = {r"m(\theta)": formula(m), \
                   r"s(\theta)": formula(s), \
                   r"a(\theta)": formula(a)}
    # ...

    return definitions
# ...

# ...
def glt_latex_names():
    """
    returns latex names for basis and atoms
    """
    # ...
    dim = 3

    symbol_names = {}
    # ...

    # ... rename basis
    B = "N"
    for i in ["i","j"]:
        Bi = B + i
        symbol_names[Symbol(Bi)] = B + "_" + i
    # ...

    # ... rename basis derivatives in the logical domain
    args_x = ARGS_x[:dim]
    args_u = ARGS_u[:dim]
    B = "N"
    for i in ["i","j"]:
        Bi = B + i
        for u in args_u + args_x:
            Bi_u = Bi + "_" + u
            partial = "\partial_" + u
            symbol_names[Symbol(Bi_u)] = partial + B + "_" + i
    # ...

    # ... rename the tensor basis derivatives
    B = "N"
    for i in ["i","j"]:
        Bi = B + i
        for k in range(0, dim):
            for s in ["", "s", "ss"]:
                Bk  = Bi + str(k+1)
                _Bk = B + "_{" + i + "_" + str(k+1) + "}"

                if len(s) > 0:
                    prime = len(s) * "\prime"

                    Bk += "_" + s
                    _Bk = B + "^{" + prime + "}" \
                            + "_{" + i + "_" + str(k+1) + "}"

                symbol_names[Symbol(Bk)] = _Bk
    # ...

    # ... TODO add flag to choose which kind of printing:
#    for k in range(0, dim):
#        # ...
#        symbol_names[Symbol('m'+str(k+1))] = "\mathfrak{m}_" + str(k+1)
#        symbol_names[Symbol('s'+str(k+1))] = "\mathfrak{s}_" + str(k+1)
#        symbol_names[Symbol('a'+str(k+1))] = "\mathfrak{a}_" + str(k+1)
#        # ...

    degree = "p"
    for k in range(0, dim):
        # ...
        for s in ["m", "s", "a"]:
            symbol_names[Symbol(s+str(k+1))] = r"\mathfrak{" + s + "}_" \
                                             + degree \
                                             + r"(\theta_" \
                                             + str(k+1) + ")"
        # ...
    # ...

    # ...
    for k in range(0, dim):
        symbol_names[Symbol("t"+str(k+1))] = r"\theta_" + str(k+1)
    # ...

    return symbol_names
# ...

# ...
def get_sympy_printer_settings(settings):
    """
    constructs the dictionary for sympy settings needed for the printer.

    settings: dict
        dictionary for different settings
    """
    sets = {}
    for key, value in list(settings.items()):
        if key not in SETTINGS:
            sets[key] = value
    return sets
# ...

# ...
def glt_latex(expr, **settings):
    """
    returns the latex expression of expr.

    expr: sympy.Expression
        a sympy expression

    settings: dict
        dictionary for different settings
    """
    # ...
    if type(expr) == dict:
        d_expr = {}
        try:
            mode = settings["mode"]
        except:
            mode = "plain"

        sets = settings.copy()
        sets["mode"] = "plain"
        for key, txt in list(expr.items()):
            d_expr[key] = glt_latex(txt, **sets)

        return d_expr
    # ...

    # ...
    try:
        from gelato.expression import glt_formatting
        fmt = settings["glt_formatting"]
        if fmt:
            expr = glt_formatting(expr, **settings)
    except:
        pass
    # ...

    # ...
    try:
        smp = settings["glt_simplify"]
        if smp:
            expr = simplify(expr)
    except:
        pass
    # ...

    # ...
    sets = get_sympy_printer_settings(settings)
    # ...

    return latex(expr, symbol_names=glt_latex_names(), **sets)
# ...

# ...
def print_glt_latex(expr, **settings):
    """
    Prints the latex expression of expr.

    settings: dict
        dictionary for different settings
    """
    print((glt_latex(expr, **settings)))
# ...


#####################################
if __name__ == '__main__':
    def test_1():
        # ... a discretization is defined as a dictionary
        discretization = {"n_elements": [16, 16, 16], \
                          "degrees": [3, 3, 3]}
        # ...

        # ... create a glt symbol from a string without evaluation
        expr = "Ni * Nj + Ni_x * Nj_x + Ni_y * Nj_y + Ni_z * Nj_z"
        expr = glt_symbol(expr, \
                          dim=3, \
                          discretization=discretization, \
                          evaluate=False)
        # ...
        print expr

    def test_2():
        # ... a discretization is defined as a dictionary
        discretization = {"n_elements": [16, 16, 16], \
                          "degrees": [3, 3, 3]}
        # ...

        from sympy import symbols
        u, v = symbols('u v')
        f = Lambda((u,v), u*v)

        # ... create a glt symbol from a string without evaluation
        expr = glt_symbol(f, \
                          dim=3, \
                          discretization=discretization, \
                          evaluate=False)
        # ...

        print expr

    test_1()
    test_2()
