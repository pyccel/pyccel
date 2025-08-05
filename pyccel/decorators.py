#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

"""
This module contains all the provided decorator methods.
"""
import warnings

__all__ = (
    'allow_negative_index',
    'elemental',
    'inline',
    'private',
    'pure',
    'stack_array',
    'types',
)


def pure(f):
    """
    Add the pure keyword to the generated Fortran function.

    Pure functions in Fortran are free from side effects. In other words, they
    can't modify global objects or the arguments of the function, and they must
    always produce the same output for the same inputs. This enables compiler
    optimisations like function reordering or parallelisation.

    Parameters
    ----------
    f : Function
        The function to which the decorator is applied.

    Returns
    -------
    Function
        The unchanged function.
    """
    return f

def private(f):
    """
    Indicate that a function shouldn't be exposed in the Python interface.

    Indicate that a function shouldn't be exposed in the Python interface. It
    is translated and can be called by other functions in the module but will
    not be wrapped.

    Parameters
    ----------
    f : Function
        The function to which the decorator is applied.

    Returns
    -------
    Function
        The unchanged function.
    """
    return f

def elemental(f):
    """
    Indicate that the function can also be used element-wise.

    This decorator indicates that the function can be applied element-wise to
    array arguments. The function operates on scalar inputs and the wrapper
    will be provided for the annotated arguments, but in the low-level code
    the function can be called on arrays whose data type matches the arguments.

    Parameters
    ----------
    f : Function
        The function to which the decorator is applied.

    Returns
    -------
    Function
        The unchanged function.
    """
    return f

def inline(f):
    """
    Indicate that the function should be inlined in the low-level code.

    This decorator indicates that the function should be inlined where it is
    called. Calls are replaced with the function body, where the
    arguments are substituted for the function parameters.

    Parameters
    ----------
    f : Function
        The function to which the decorator is applied.

    Returns
    -------
    Function
        The unchanged function.
    """
    return f

def stack_array(*args):
    """
    Indicate that arrays should be stored on the stack.

    This decorator indicates that all arrays whose names appear among the
    arguments should be stored on the stack.

    Parameters
    ----------
    *args : str
        The names of all arrays which should be stored on the stack.

    Returns
    -------
    Function
        The identity decorator which will be applied to the function.
    """
    def identity(f):
        return f
    return identity

def allow_negative_index(*args):
    """
    Indicate that arrays can be accessed with negative indexes.

    This decorator indicates that all arrays mentioned as args can be accessed with
    negative indexes. As a result all non-constant indexing uses a modulo
    function. This can have negative results on the performance.

    Parameters
    ----------
    *args : str
        The names of all arrays which can be accessed with non-constant
        negative indexes.

    Returns
    -------
    Function
        The identity decorator which will be applied to the function.
    """
    def identity(f):
        return f
    return identity
