#!/usr/bin/python
# -*- coding: utf-8 -*-

from sympy.core.function import Function
from .core import DottedName
from .core import Import
from .core import Range, Len
from .core import FunctionDef, Return
from .numpyext import Zeros, Ones
from .numpyext import Array, Shape, Int, Sum, Rand
from sympy import (Abs, sqrt, sin, cos, exp, log, csc, cos, sec, tan, cot, asin,
                   acsc, acos, asec, atan, acot, atan2, Mod, Max, Min)

math_functions = {
    'Abs': Abs,
    'sqrt': sqrt,
    'sin': sin,
    'cos': cos,
    'exp': exp,
    'log': log,
    'csc': csc,
    'sec': sec,
    'tan': tan,
    'cot': cot,
    'asin': asin,
    'acsc': acsc,
    'acos': acos,
    'asec': asec,
    'atan': atan,
    'acot': acot,
    'atan2': atan2,
    }


def builtin_function(expr, args=None):
    """Returns a builtin-function call applied to given arguments."""

    if not (isinstance(expr, Function) or isinstance(expr, str)):
        raise TypeError('Expecting a string or a Function class')

    if isinstance(expr, Function):
        name = str(type(expr).__name__)

    if isinstance(expr, str):
        name = expr

    if name == 'range':
        return Range(*args)
    if name == 'array':
        return Array(*args)
    if name == 'int':
        return Int(*args)
    if name == 'len':
        return Len(*args)
    if name == 'sum':
        return Sum(*args)
    if name == 'Mod':
        return Mod(*args)
    if name == 'Max':
        return Max(*args)
    
    if name == 'lambdify':
       code = compile(args.body[0],'','single')
       g={} 
       eval(code,g)
       f_name = str(args.name)
       code = g[f_name]
       args_ = args.arguments
       expr = code(*args_)
       f_arguments = list(expr.free_symbols)
       func = FunctionDef(f_name, f_arguments, [], [Return(expr)],decorators = args.decorators)
       return func

    return None

# TODO add documentation
builtin_import_registery = ('numpy', 'scipy', 'itertools')

def builtin_import(expr):
    """Returns a builtin pyccel-extension function/object from an import."""

    if not isinstance(expr, Import):
        raise TypeError('Expecting an Import expression')

    if expr.source is None:
        return (None, None)

    source = expr.source
    if isinstance(source, DottedName):
        source = source.name[0]

        # TODO imrove

    if source == 'numpy':

        # TODO improve

        target = str(expr.target[0])
        if target == 'zeros':

            # TODO return as_name and not name

            return (target, Zeros)

        if target == 'ones':

            # TODO return as_name and not name

            return (target, Ones)

        if target == 'array':
            return (target, Array)

        if target == 'shape':
            return (target, Shape)

        if target == 'int':
            return (target, Int)

        if target == 'sum':
            return (target, Sum)

        if target in ['rand', 'random']:
            return (target, Rand)

        if target in math_functions.keys():
            return (target, math_functions[target])
    elif source == 'math':

        target = str(expr.target[0])
        if target in math_functions.keys():
            return (target, math_functions[target])

    return (None, None)

