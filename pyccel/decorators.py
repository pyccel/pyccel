#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This module contains all the provided decorator methods.
"""
import numpy

#TODO use pycode and call exec after that in lambdify

__all__ = (
    'allow_negative_index',
    'bypass',
    'elemental',
    'inline',
    'lambdify',
    'private',
    'pure',
    'stack_array',
    'sympy',
    'template',
    'types',
    'kernel',
)

def lambdify(f):

    args = f.__code__.co_varnames
    from sympy import symbols
    args = symbols(args)
    expr = f(*args)
    def wrapper(*vals):
        return  expr.subs(zip(args,vals)).doit()

    return wrapper

def sympy(f):
    return f

def bypass(f):
    return f

def types(*args,**kw):
    def identity(f):
        return f
    return identity

def template(name, types=()):
    """template decorator."""
    def identity(f):
        return f
    return identity

def pure(f):
    return f

def private(f):
    return f

def elemental(f):
    return f

def inline(f):
    """Indicates that function calls to this function should
    print the function body directly"""
    return f

def stack_array(f, *args):
    """
    Decorator indicates that all arrays mentioned as args should be stored
    on the stack.

    Parameters
    ----------
    f : Function
        The function to which the decorator is applied
    args : list of str
        A list containing the names of all arrays which should be stored on the stack
    """
    def identity(f):
        return f
    return identity

def allow_negative_index(f,*args):
    """
    Decorator indicates that all arrays mentioned as args can be accessed with
    negative indexes. As a result all non-constant indexing uses a modulo
    function. This can have negative results on the performance

    Parameters
    ----------
    f : Function
        The function to which the decorator is applied
    args : list of str
        A list containing the names of all arrays which can be accessed
        with non-constant negative indexes
    """
    def identity(f):
        return f
    return identity

def kernel(f):
    return numpy.array([[f]])