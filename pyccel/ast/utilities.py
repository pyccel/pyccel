#!/usr/bin/python
# -*- coding: utf-8 -*-

import inspect

from sympy.core.function import Application
from sympy import Not, Function
from numpy import pi

import pyccel.decorators as pyccel_decorators
from pyccel.symbolic import lambdify
from pyccel.errors.errors import Errors

from .core     import (AsName, Import, Product, FunctionDef, String, Constant,
                       Variable, IndexedVariable, ValuedVariable)

from .builtins import (PythonBool, Enumerate, PythonInt, PythonFloat,
                       PythonComplex, Len, Map, Range, Zip)

from .mathext  import math_functions, math_constants

from .numpyext import (NumpyFull, NumpyEmpty, NumpyZeros, NumpyOnes, NumpyDiag,
                       NumpyFullLike, NumpyEmptyLike, NumpyZerosLike, NumpyExp,
                       NumpyMin, NumpyMax, NumpyAbs, NumpyFloor, NumpyNorm,
                       NumpyArray, NumpyRand, NumpyMatmul, NumpySum, NumpyReal,
                       NumpyInt, NumpyInt32, NumpyInt64, NumpyFloat, Shape,
                       NumpyRandint, NumpyLog, NumpySqrt, NumpyComplex,
                       NumpySin, NumpyCos, NumpyTan, NumpyComplex64, NumpyMod,
                       NumpyArcsin, NumpyArccos, NumpyArctan, NumpyArctan2,
                       NumpySinh, NumpyCosh, NumpyTanh, NumpyFabs, NumpyImag,
                       NumpyArcsinh, NumpyArccosh, NumpyArctanh, NumpyFloat32,
                       numpy_constants, NumpyLinspace, NumpyOnesLike,
                       NumpyProduct, NumpyFloat64, NumpyWhere, NumpyComplex128)

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
    'full'      : NumpyFull,
    'empty'     : NumpyEmpty,
    'zeros'     : NumpyZeros,
    'ones'      : NumpyOnes,
    'full_like' : NumpyFullLike,
    'empty_like': NumpyEmptyLike,
    'zeros_like': NumpyZerosLike,
    'ones_like' : NumpyOnesLike,
    'array'     : NumpyArray,
    # ...
    'shape'     : Shape,
    'norm'      : NumpyNorm,
    'int'       : NumpyInt,
    'real'      : NumpyReal,
    'imag'      : NumpyImag,
    'float'     : NumpyFloat,
    'double'    : NumpyFloat64,
    'mod'       : NumpyMod,
    'float32'   : NumpyFloat32,
    'float64'   : NumpyFloat64,
    'int32'     : NumpyInt32,
    'int64'     : NumpyInt64,
    'complex'   : NumpyComplex,
    'complex128': NumpyComplex128,
    'complex64' : NumpyComplex64,
    'matmul'    : NumpyMatmul,
    'sum'       : NumpySum,
    'max'      : NumpyMax,
    'min'      : NumpyMin,
    'prod'      : NumpyProduct,
    'product'   : NumpyProduct,
    'linspace'  : NumpyLinspace,
    'diag'      : NumpyDiag,
    'where'     : NumpyWhere,
    # 'cross'     : NumpyCross,   # Currently not correctly implemented
    # ---
    'abs'       : NumpyAbs,
    'floor'     : NumpyFloor,
    'absolute'  : NumpyAbs,
    'fabs'      : NumpyFabs,
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
    # 'hypot'     : NumpyHypot,
    'sinh'      : NumpySinh,
    'cosh'      : NumpyCosh,
    'tanh'      : NumpyTanh,
    'arcsinh'   : NumpyArcsinh,
    'arccosh'   : NumpyArccosh,
    'arctanh'   : NumpyArctanh,
    # 'deg2rad'   : NumpyDeg2rad,
    # 'rad2deg'   : NumpyRad2deg,
}

numpy_linalg_functions = {
    'norm'      : NumpyNorm,
}

numpy_random_functions = {
    'rand'      : NumpyRand,
    'random'    : NumpyRand,
    'randint'   : NumpyRandint,
}

builtin_functions_dict = {
    'abs'      : NumpyAbs,  # TODO: create a built-in Abs
    'range'    : Range,
    'zip'      : Zip,
    'enumerate': Enumerate,
    'int'      : PythonInt,
    'float'    : PythonFloat,
    'complex'  : PythonComplex,
    'bool'     : PythonBool,
    'sum'      : NumpySum,
    'len'      : Len,
    'max'      : NumpyMax,
    'min'      : NumpyMin,
    'not'      : Not,   # TODO [YG, 20.05.2020]: do not use Sympy's Not
}

scipy_constants = {
    'pi': Constant('real', 'pi', value=pi),
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
builtin_import_registery = {'numpy': {
                                      **numpy_functions,
                                      **numpy_constants,
                                      'linalg':numpy_linalg_functions,
                                      'random':numpy_random_functions
                                      },
                            'numpy.linalg': numpy_linalg_functions,
                            'numpy.random': numpy_random_functions,
                            'scipy.constants': scipy_constants,
                            'itertools': {'product': Product},
                            'math': {**math_functions, ** math_constants},
                            'pyccel.decorators': None}

#==============================================================================
def collect_relevant_imports(func_dictionary, targets):
    if len(targets) == 0:
        return func_dictionary

    imports = []
    for target in targets:
        if isinstance(target, AsName):
            import_name = target.name
            code_name = target.target
        else:
            import_name = str(target)
            code_name = import_name

        if import_name in func_dictionary.keys():
            imports.append((code_name, func_dictionary[import_name]))
    return imports

def builtin_import(expr):
    """Returns a builtin pyccel-extension function/object from an import."""

    if not isinstance(expr, Import):
        raise TypeError('Expecting an Import expression')

    if isinstance(expr.source, AsName):
        source = str(expr.source.name)
    else:
        source = str(expr.source)

    if source == 'pyccel.decorators':
        funcs = [f[0] for f in inspect.getmembers(pyccel_decorators, inspect.isfunction)]
        for target in expr.target:
            search_target = target.name if isinstance(target, AsName) else str(target)
            if search_target not in funcs:
                errors = Errors()
                errors.report("{} does not exist in pyccel.decorators".format(target),
                        symbol = expr, severity='error')

    elif source in builtin_import_registery:
        return collect_relevant_imports(builtin_import_registery[source], expr.target)

    return []

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

        dtype = String(dtype)
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
