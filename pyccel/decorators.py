#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This module contains all the provided decorator methods.
"""
import warnings

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

def types(*args, results = None):
    """
    Specify the types passed to the function.

    Specify the types passed to the function.

    Parameters
    ----------
    *args : tuple of str or types
        The types of the arguments of the function.

    results : str or type, optional
        The return type of the function.

    Returns
    -------
    decorator
        The identity decorator which will not modify the function.
    """
    warnings.warn("The @types decorator will be removed in a future version of " +
                  "Pyccel. Please use type hints. The @template decorator can be " +
                  "used to specify multiple types. See the documentation at " +
                  "https://github.com/pyccel/pyccel/blob/devel/docs/quickstart.md#type-annotations"
                  "for examples.", FutureWarning)
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
