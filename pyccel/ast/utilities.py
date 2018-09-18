#!/usr/bin/python
# -*- coding: utf-8 -*-

from sympy.core.function import Application
from .core import DottedName
from .core import Import
from .core import Range, Len , Enumerate, Zip, Product, Map
from .core import FunctionDef, Return, Assign
from .core import Constant,ZerosLike
from .numpyext import Zeros, Ones, Empty
from .numpyext import Array, Shape, Int, Rand, Sum, Real, Complex
from .numpyext import Int64, Int32, Float32, Float64, Complex64, Complex128
from .numpyext import Sqrt, Asin, Acsc, Acos, Asec, Atan, Acot, Sinh, Cosh, Tanh, Log
from .numpyext import numpy_constants
from sympy import Symbol, Lambda, floor
from sympy import Not,Float
from sympy import Function
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
    'acot': Acot,
    'sinh': Sinh,
    'cosh': Cosh,
    'tanh': Tanh
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

    if name == 'range':
        return Range(*args)
    elif name == 'zip':
        return Zip(*args)
    elif name == 'enumerate':
        return Enumerate(*args)
    if name == 'array':
        return Array(*args)
    if name in ['int']:
        return Int(*args)
    if name in ['float']:
        return Real(*args)
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
                imports.append((target, Int))
  
            elif target == 'real':
                imports.append((target, Real))

            elif target == 'float32':
                imports.append((target, Float32))

            elif target == 'float64':
                imports.append((target, Float64))

            elif target == 'int64':
                imports.append((target, Int64))

            elif target == 'int32':
                imports.append((target, Int32))

            elif target == 'complex128':
                imports.append((target, Complex128))

            elif target == 'complex64':
                imports.append((target, Complex64))

            elif target == 'sum':
                imports.append((target,Sum))

            elif target in ['rand', 'random']:
                imports.append((target, Rand))

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
