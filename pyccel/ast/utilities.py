#!/usr/bin/python
# -*- coding: utf-8 -*-

from sympy.core.function import Application
from .core import DottedName
from .core import Import
from .core import Range, Len , Enumerate, Zip, Product, Map
from .core import FunctionDef, Return, Assign

from .core import Constant, Variable, IndexedVariable
from .numpyext import Zeros, Ones, Empty, ZerosLike, FullLike, Diag, Cross
from .numpyext import Min, Max, Abs, Norm, EmptyLike, Where
from .numpyext import Array, Shape, Int, Rand, Sum, Real, Complex, Imag, Mod
from .numpyext import Int64, Int32, Float32, Float64, Complex64, Complex128
from .numpyext import Sqrt, Asin, Acsc, Acos, Asec, Atan, Acot, Sinh, Cosh, Tanh, Log
from .numpyext import numpy_constants, Linspace
from pyccel.symbolic import lambdify
from sympy import Symbol, Lambda, floor
from sympy import Not, Float
from sympy import Function
from sympy import (sin, cos, exp, csc, cos, sec, tan, cot, atan2)

import scipy.constants as sc_constants

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
    'zeros'     : Zeros,
    'empty'     : Empty,
    'ones'      : Ones,
    'zeros_like': ZerosLike,
    'empty_like': EmptyLike,
    'full_like' : FullLike,
    'array'     : Array,
    # ...
    'shape'     : Shape,
    'norm'      : Norm,
    'int'       : Int,
    'real'      : Real,
    'imag'      : Imag,
    'float'     : Real,
    'double'    : Real,
    'Mod'       : Mod,
    'float32'   : Float32,
    'float64'   : Float64,
    'int32'     : Int32,
    'int64'     : Int64,
    'complex128': Complex128,
    'complex64' : Complex64,
    'sum'       : Sum,
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
    'float'    : Real,
    'sum'      : Sum,
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


def builtin_function(expr, args=None):
    """Returns a builtin-function call applied to given arguments."""
    if not (isinstance(expr, Application) or isinstance(expr, str)):
        raise TypeError('Expecting a string or a Function class')

    if isinstance(expr, Application):
        name = str(type(expr).__name__)
    elif isinstance(expr, str):
        name = expr
    else:
        raise TypeError('expr must be of type str or Function')

    dic = builtin_functions_dict

    if name in dic.keys() :
        return dic[name](*args)
    elif name == 'array':
        return Array(*args)
    elif name in ['complex']:
        if len(args)==1:
            args = [args[0],Float(0)]
        return Complex(args[0],args[1])
    elif name == 'Not':
        return Not(*args)

    elif name == 'map':
        func = Function(str(expr.args[0].name))
        args = [func]+list(args[1:])
        return Map(*args)

    elif name == 'lambdify':
        return lambdify(expr, args)

    return None

# TODO add documentation
builtin_import_registery = ('numpy', 'scipy', 'itertools', 'math')

def builtin_import(expr):
    """Returns a builtin pyccel-extension function/object from an import."""

    if not isinstance(expr, Import):
        raise TypeError('Expecting an Import expression')

    if expr.source is None:
        return []

    source = expr.source
    if isinstance(source, DottedName):
        source = source.name[0]
    else:
        source = str(source)

        # TODO imrove
    imports = []
    for i in range(len(expr.target)):
        if source == 'numpy':

            target = str(expr.target[i])
            if target in numpy_functions.keys():
                imports.append((target, numpy_functions[target]))

            elif target in math_functions.keys():
                imports.append((target, math_functions[target]))

            elif target in numpy_constants.keys():
                imports.append((target, numpy_constants[target]))

        elif source == 'math':

            target = str(expr.target[i])

            if target in math_functions.keys():
                imports.append((target, math_functions[target]))

        elif source == 'scipy':
            # TODO improve: source must be scipy.constants
            #      - use dynamic import?
            target = str(expr.target[i])
            if target in scipy_constants.keys():
                imports.append((target, scipy_constants[target]))
        elif source == 'itertools':
            target = str(expr.target[i])

            if target == 'product':
                imports.append((target, Product))



    return imports

def get_function_from_ast(ast, func_name):
    node = None
    for stmt in ast:
        if isinstance(stmt, FunctionDef) and str(stmt.name) == func_name:
            node = stmt
            break

    if node is None:
        print('> could not find {}'.format(func_name))

    return node

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
