# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This module provides us with functions and objects that allow us to compute
the computational intensity

Example
-------

"""

from sympy import sympify, Symbol
from sympy import Poly, LT

from pyccel.complexity.arithmetic import OpComplexity
from pyccel.complexity.memory import MemComplexity

from pyccel.complexity.arithmetic import ADD, SUB, MUL, DIV, IDIV, ABS
from pyccel.complexity.memory import READ, WRITE

_cost_symbols = {ADD, SUB, MUL, DIV, IDIV, ABS,
                 READ, WRITE}

__all__ = ["computational_intensity"]


def _leading_term(expr, *args):
    """
    Returns the leading term in a sympy Polynomial.

    expr: sympy.Expr
        any sympy expression

    args: list
        list of input symbols for univariate/multivariate polynomials
    """
    expr = sympify(str(expr))
    P = Poly(expr, *args)
    return LT(P)
# ...

# ...
def computational_intensity(filename_or_text, args=None, mode=None,
                            verbose=False):

    # ...
    complexity = OpComplexity(filename_or_text)
    f = complexity.cost(mode=mode)

    complexity = MemComplexity(filename_or_text)
    m = complexity.cost(mode=mode)
    # ...

    # ...
    args = f.free_symbols.union(m.free_symbols) - _cost_symbols
    args = list(args)
    # ...

    # ...
    lt_f = _leading_term(f, *args)
    lt_m = _leading_term(m, *args)

    q = lt_f/lt_m
    # ...

    # ...
    if verbose:
        print((" arithmetic cost         ~ " + str(f)))
        print((" memory cost             ~ " + str(m)))
        print((" computational intensity ~ " + str(q)))
    # ...

    return q
