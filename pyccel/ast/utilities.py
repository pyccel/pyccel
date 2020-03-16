#!/usr/bin/python
# -*- coding: utf-8 -*-

from sympy.core.function import Application
from sympy import Symbol, Lambda, floor
from sympy import Not, Float
from sympy import Function
from sympy import (sin, cos, exp, csc, cos, sec, tan, cot, atan2)
import scipy.constants as sc_constants

from pyccel.symbolic import lambdify

from .core import AsName
from .core import Import
from .core import Product
from .core import FunctionDef, Return, Assign
from .core import ValuedArgument
from .core import Constant, Variable, IndexedVariable

from .builtins import Bool, Enumerate, Int, PythonFloat, Len, Map, Range, Zip

from .numpyext import Full, Empty, Zeros, Ones
from .numpyext import FullLike, EmptyLike, ZerosLike, OnesLike
from .numpyext import Diag, Cross
from .numpyext import Min, Max, Abs, Norm, Where
from .numpyext import Array, Shape, Rand, NumpySum, Matmul, Real, Complex, Imag, Mod
from .numpyext import NumpyInt, Int32, Int64, NumpyFloat, Float32, Float64, Complex64, Complex128
from .numpyext import Sqrt, Asin, Acsc, Acos, Asec, Atan, Acot, Sinh, Cosh, Tanh, Log
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
math_functions = {
    'abs'    : Abs,
    'sqrt'   : Sqrt,
    'sin'    : sin,
    'cos'    : cos,
    'exp'    : exp,
    'log'    : Log,
    'csc'    : csc,
    'sec'    : sec,
    'tan'    : tan,
    'cot'    : cot,
    'asin'   : Asin,
    'acsc'   : Acsc,
    'arccos' : Acos,
    'acos'   : Acos,
    'asec'   : Asec,
    'atan'   : Atan,
    'acot'   : Acot,
    'sinh'   : Sinh,
    'cosh'   : Cosh,
    'tanh'   : Tanh,
    'atan2'  : atan2,
    'arctan2': atan2
    }

# TODO split numpy_functions into multiple dictionaries following
# https://docs.scipy.org/doc/numpy-1.15.0/reference/routines.array-creation.html
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
    'complex128': Complex128,
    'complex64' : Complex64,
    'matmul'    : Matmul,
    'sum'       : NumpySum,
    'prod'      : Prod,
    'product'   : Prod,
    'rand'      : Rand,
    'random'    : Rand,
    'linspace'  : Linspace,
    'diag'      : Diag,
    'where'     : Where,
    'cross'     : Cross,
}

builtin_functions_dict = {
    'range'    : Range,
    'zip'      : Zip,
    'enumerate': Enumerate,
    'int'      : Int,
    'float'    : PythonFloat,
    'bool'     : Bool,
    'sum'      : NumpySum,
    'len'      : Len,
    'Mod'      : Mod,
    'abs'      : Abs,
    'max'      : Max,
    'Max'      : Max,
    'min'      : Min,
    'Min'      : Min,
    'floor'    : floor,
    'not'      : Not
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

    if name in ['complex']:
        if len(args)==1:
            args = [args[0], Float(0)]
        return Complex(args[0],args[1])

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
builtin_import_registery = ('numpy', 'scipy.constants', 'itertools', 'math')

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

            elif import_name in math_functions.keys():
                imports.append((code_name, math_functions[import_name]))

            elif import_name in numpy_constants.keys():
                imports.append((code_name, numpy_constants[import_name]))

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
def get_external_function_from_ast(ast):
    nodes   = []
    others  = []
    for stmt in ast:
        if isinstance(stmt, FunctionDef):
            if stmt.is_external or stmt.is_external_call:
                nodes += [stmt]

            else:
                others += [stmt]

    return nodes, others

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
        if isinstance(a, ValuedArgument):
            args, val_args = args[:i], args[i:]
            break

    # Convert list of keyword arguments into dictionary
    kwargs = {}
    for v in val_args:
        key   = str(v.argument.name)
        value = v.value
        kwargs[key] = value

    return args, kwargs
