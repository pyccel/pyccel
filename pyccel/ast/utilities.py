# coding: utf-8

from sympy.core.function import Function

from .core import Import
from .core import Range
from .numpyext import Zeros, Ones
from .core import Array

def builtin_function(expr, args=None):
    """Returns a builtin-function call applied to given arguments."""
    if not(isinstance(expr, Function) or isinstance(expr, str)):
        raise TypeError('Expecting a string or a Function class')

    if isinstance(expr, Function):
        name = str(type(expr).__name__)

    if isinstance(expr, str):
        name = expr

    if name == 'range':
        return Range(*args)
    elif name == 'array':
        return Array(args)

    return None

def builtin_import(expr):
    """Returns a builtin pyccel-extension function/object from an import."""
    if not isinstance(expr, Import):
        raise TypeError('Expecting an Import expression')

    if expr.source is None:
        return None, None

    source = expr.source
    if source == 'numpy':
        # TODO improve
        target = str(expr.target[0])
        if target == 'zeros':
            # TODO return as_name and not name
            return target, Zeros

        if target == 'ones':
            # TODO return as_name and not name
            return target, Ones

    return None, None
