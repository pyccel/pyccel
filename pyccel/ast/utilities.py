#!/usr/bin/python
# -*- coding: utf-8 -*-

from sympy.core.function import Application
from .core import DottedName
from .core import Import
from .core import Range, Len , Enumerate, Zip, Product
from .core import FunctionDef, Return, Assign
from .core import Constant,ZerosLike
from .numpyext import Zeros, Ones, Empty
from .numpyext import Array, Shape, Int, Rand, Sum
from .numpyext import Sqrt, Asin, Acsc, Acos, Asec, Atan, Acot, Log
from sympy import Symbol, Lambda, floor
from sympy import I
from sympy import Not
from sympy import (Abs, sin, cos, exp, csc, cos, sec, tan, cot, Mod, Max, Min)

import scipy.constants as sc_constants

math_functions = {
    'abs': Abs,
    'sqrt': Sqrt,
    'sin': sin,
    'cos': cos,
    'exp': exp,
    'log': Log,
    'csc': csc,
    'sec': sec,
    'tan': tan,
    'cot': cot,
    'asin': Asin,
    'acsc': Acsc,
    'arccos': Acos,
    'acos':Acos,
    'asec': Asec,
    'atan': Atan,
    'acot': Acot
    }

scipy_constants = {
    'pi': Constant('double', 'pi', value=sc_constants.pi),
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

    if name == 'range':
        return Range(*args)
    elif name == 'zip':
        return Zip(*args)
    elif name == 'enumerate':
        return Enumerate(*args)
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
    if name == 'abs':
        return Abs(*args)
    if name in ['max', 'Max']:
        return Max(*args)
    if name in ['min', 'Min']:
        return Min(*args)
    if name == 'floor':
        return floor(*args)
    elif name == 'complex':
        return args[0]+I*args[1]
    elif name == 'Not':
        return Not(*args)
    

    if name == 'lambdify':
       if isinstance(args, Lambda):
           expr_ = args.expr
           expr_ = Return(expr_)
           expr_.set_fst(expr)
           f_arguments = args.variables
           func = FunctionDef('lambda', f_arguments, [], [expr_])
           return func
           
       code = compile(args.body[0],'','single')
       g={}
       eval(code,g)
       f_name = str(args.name)
       code = g[f_name]
       args_ = args.arguments
       expr_ = code(*args_)
       f_arguments = list(expr_.free_symbols)
       expr_ = Return(expr_)
       expr_.set_fst(expr)
       body = [expr_]
       func = FunctionDef(f_name, f_arguments, [], body ,decorators = args.decorators)
       return func

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

        # TODO improve

            target = str(expr.target[i])
            if target == 'zeros':
                imports.append((target, Zeros))

            elif target == 'ones':
                imports.append((target, Ones))

            elif target == 'empty':
                imports.append((target, Empty))
   
            elif target == 'zeros_like':
                imports.append((target,ZerosLike))

            elif target == 'array':
                imports.append((target, Array))

            elif target == 'shape':
                imports.append((target, Shape))

            elif target == 'int':
                imports.append((target,Int))

            elif target == 'sum':
                imports.append((target,Sum))

            elif target in ['rand', 'random']:
                imports.append((target, Rand))

            elif target in math_functions.keys():
                imports.append((target, math_functions[target]))

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
