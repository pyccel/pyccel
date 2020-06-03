#!/usr/bin/python
# -*- coding: utf-8 -*-

from sympy.core.function import Application
from sympy import Not, Float
from sympy import Function
import scipy.constants as sc_constants

from pyccel.symbolic import lambdify

from .core import AsName
from .core import Import
from .core import Product
from .core import FunctionDef
from .core import ValuedVariable
from .core import Constant, Variable, IndexedVariable

from .builtins import Bool, Enumerate, Int, PythonFloat, PythonComplex, Len, Map, Range, Zip

from .mathext  import math_functions

from .numpyext import Full, Empty, Zeros, Ones
from .numpyext import FullLike, EmptyLike, ZerosLike, OnesLike
from .numpyext import Diag, Cross
from .numpyext import Min, Max, NumpyAbs, NumpyFloor, Norm, Where
from .numpyext import Array, Shape, Rand, NumpySum, Matmul, Real, NumpyComplex, Imag, Mod
from .numpyext import NumpyInt, Int32, Int64, NumpyFloat, Float32, Float64, Complex64, Complex128
from .numpyext import NumpyExp, NumpyLog, NumpySqrt
from .numpyext import NumpySin, NumpyCos, NumpyTan
from .numpyext import NumpyArcsin, NumpyArccos, NumpyArctan, NumpyArctan2
from .numpyext import NumpySinh, NumpyCosh, NumpyTanh
from .numpyext import NumpyArcsinh, NumpyArccosh, NumpyArctanh
from .numpyext import numpy_constants, Linspace
from .numpyext import Product as Prod


__all__ = (
    'build_types_decorator',
    'builtin_function',
    'builtin_import',
    'builtin_import_registery',
    'split_positional_keyword_arguments',
)

#==============================================================================
# TODO split numpy_functions into multiple dictionaries following
# https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.array-creation.html
# TODO [YG, 20.05.2020]: Move dictionary to 'numpyext' module
numpy_functions = {
    # ... array creation routines
    'full'      : Full,
    'empty'     : Empty,
    'zeros'     : Zeros,
    'ones'      : Ones,
    'full_like' : FullLike,
    'empty_like': EmptyLike,
    'zeros_like': ZerosLike,
    'ones_like' : OnesLike,
    'array'     : Array,
    # ...
    'shape'     : Shape,
    'norm'      : Norm,
    'int'       : NumpyInt,
    'real'      : Real,
    'imag'      : Imag,
    'float'     : NumpyFloat,
    'double'    : Float64,
    'mod'       : Mod,
    'float32'   : Float32,
    'float64'   : Float64,
    'int32'     : Int32,
    'int64'     : Int64,
    'complex'   : NumpyComplex,
    'complex128': Complex128,
    'complex64' : Complex64,
    'matmul'    : Matmul,
    'sum'       : NumpySum,
    'prod'      : Prod,
    'product'   : Prod,
    'linspace'  : Linspace,
    'diag'      : Diag,
    'where'     : Where,
    'cross'     : Cross,
    # ---
    'abs'       : NumpyAbs,
    'floor'     : NumpyFloor,
    'absolute'  : NumpyAbs,
    'fabs'      : NumpyAbs,
    'exp'       : NumpyExp,
    'log'       : NumpyLog,
    'sqrt'      : NumpySqrt,
    # ---
    'sin'       : NumpySin,
    'cos'       : NumpyCos,
    'tan'       : NumpyTan,
    'arcsin'    : NumpyArcsin,
    'arccos'    : NumpyArccos,
    'arctan'    : NumpyArctan,
    'arctan2'   : NumpyArctan2,
#    'hypot'     : NumpyHypot,
    'sinh'      : NumpySinh,
    'cosh'      : NumpyCosh,
    'tanh'      : NumpyTanh,
    'arcsinh'   : NumpyArcsinh,
    'arccosh'   : NumpyArccosh,
    'arctanh'   : NumpyArctanh,
#    'deg2rad'   : NumpyDeg2rad,
#    'rad2deg'   : NumpyRad2deg,
}

numpy_linalg_functions = {
    'norm'      : Norm,
}

numpy_random_functions = {
    'rand'      : Rand,
    'random'    : Rand,
}

builtin_functions_dict = {
    'abs'      : NumpyAbs,  # TODO: create a built-in Abs
    'range'    : Range,
    'zip'      : Zip,
    'enumerate': Enumerate,
    'int'      : Int,
    'float'    : PythonFloat,
    'complex'  : PythonComplex,
    'bool'     : Bool,
    'sum'      : NumpySum,
    'len'      : Len,
    'Mod'      : Mod,
    'max'      : Max,
#    'Max'      : Max,
    'min'      : Min,
#    'Min'      : Min,
    'not'      : Not,   # TODO [YG, 20.05.2020]: do not use Sympy's Not
}

scipy_constants = {
    'pi': Constant('real', 'pi', value=sc_constants.pi),
                  }

#==============================================================================
def builtin_function(expr, args=None):
    """Returns a builtin-function call applied to given arguments."""

    if isinstance(expr, Application):
        name = str(type(expr).__name__)
    elif isinstance(expr, str):
        name = expr
    else:
        raise TypeError('expr must be of type str or Function')

    dic = builtin_functions_dict

    if name in dic.keys() :
        return dic[name](*args)

    if name == 'Not':
        return Not(*args)

    if name == 'map':
        func = Function(str(expr.args[0].name))
        args = [func]+list(args[1:])
        return Map(*args)

    if name == 'lambdify':
        return lambdify(expr, args)

    return None

# TODO add documentation
builtin_import_registery = ('numpy', 'numpy.linalg', 'numpy.random', 'scipy.constants', 'itertools', 'math')

#==============================================================================
def builtin_import(expr):
    """Returns a builtin pyccel-extension function/object from an import."""

    if not isinstance(expr, Import):
        raise TypeError('Expecting an Import expression')

    if expr.source is None:
        return []

    source = str(expr.source)

        # TODO imrove
    imports = []
    for target in expr.target:
        if isinstance(target, AsName):
            import_name = target.target
            code_name = target.name
        else:
            import_name = str(target)
            code_name = import_name
        if source == 'numpy':

            if import_name in numpy_functions.keys():
                imports.append((code_name, numpy_functions[import_name]))

            elif import_name in numpy_constants.keys():
                imports.append((code_name, numpy_constants[import_name]))

        elif source == 'numpy.linalg':

            if import_name in numpy_linalg_functions.keys():
                imports.append((code_name, numpy_linalg_functions[import_name]))

        elif source == 'numpy.random':

            if import_name in numpy_random_functions.keys():
                imports.append((code_name, numpy_random_functions[import_name]))

        elif source == 'math':

            if import_name in math_functions.keys():
                imports.append((code_name, math_functions[import_name]))

        elif source == 'scipy.constants':
            if import_name in scipy_constants.keys():
                imports.append((code_name, scipy_constants[import_name]))
        elif source == 'itertools':

            if import_name == 'product':
                imports.append((code_name, Product))

    return imports

#==============================================================================
def get_function_from_ast(ast, func_name):
    node = None
    for stmt in ast:
        if isinstance(stmt, FunctionDef) and str(stmt.name) == func_name:
            node = stmt
            break

    if node is None:
        print('> could not find {}'.format(func_name))

    return node

#==============================================================================
# TODO: must add a Node Decorator in core
def build_types_decorator(args, order=None):
    """
    builds a types decorator from a list of arguments (of FunctionDef)
    """
    types = []
    for a in args:
        if isinstance(a, Variable):
            dtype = a.dtype.name.lower()

        elif isinstance(a, IndexedVariable):
            dtype = a.dtype.name.lower()

        else:
            raise TypeError('unepected type for {}'.format(a))

        if a.rank > 0:
            shape = [':' for i in range(0, a.rank)]
            shape = ','.join(i for i in shape)
            dtype = '{dtype}[{shape}]'.format(dtype=dtype, shape=shape)
            if order and a.rank > 1:
                dtype = "{dtype}(order={ordering})".format(dtype=dtype, ordering=order)

        if not ( dtype.startswith("'") and dtype.endswith("'") ):
            dtype = "'{}'".format(dtype)

        types.append(dtype)

    return types

#==============================================================================
def split_positional_keyword_arguments(*args):
    """ Create a list of positional arguments and a dictionary of keyword arguments
    """

    # Distinguish between positional and keyword arguments
    val_args = ()
    for i, a in enumerate(args):
        if isinstance(a, ValuedVariable):
            args, val_args = args[:i], args[i:]
            break

    # Convert list of keyword arguments into dictionary
    kwargs = {}
    for v in val_args:
        key   = str(v.name)
        value = v.value
        kwargs[key] = value

    return args, kwargs

