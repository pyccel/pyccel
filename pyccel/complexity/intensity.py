# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

# TODO - use SHAPE(...) as args for Poly => ex1.py breaks down for a function
#        like array_int32_1d_scalar_add
#      - for the moment, we should put only tests that are working (because of
#      SHAPE(...))

"""
This module provides us with functions and objects that allow us to compute
the computational intensity

Example
-------

"""

from collections import OrderedDict

from sympy import sympify, Symbol
from sympy import Poly, LT

from pyccel.complexity.arithmetic import OpComplexity
from pyccel.complexity.memory import MemComplexity

from pyccel.complexity.arithmetic import ADD, SUB, MUL, DIV, IDIV, ABS
from pyccel.complexity.memory import READ, WRITE

_cost_symbols = {ADD, SUB, MUL, DIV, IDIV, ABS,
                 READ, WRITE}

__all__ = ["computational_intensity"]


# ==============================================================================
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

# ==============================================================================
def _intensity(f, m):
    if f * m == 0:
        return 0

    f = sympify(str(f))
    m = sympify(str(m))

    args = f.free_symbols.union(m.free_symbols) - _cost_symbols
    args = list(args)

    lt_f = _leading_term(f, *args)
    lt_m = _leading_term(m, *args)

    return lt_f/lt_m

# ==============================================================================
class ComputationalIntensity(object):
    def __init__(self, filename_or_text):
        self._arithmetic = OpComplexity(filename_or_text)
        self._memory     = MemComplexity(filename_or_text)
        self._costs = OrderedDict()

    @property
    def arithmetic(self):
        return self._arithmetic

    @property
    def memory(self):
        return self._memory

    @property
    def costs(self):
        return self._costs

    def cost(self, mode=None):
        f = self.arithmetic.cost(mode=mode)
        m = self.memory.cost(mode=mode)

        f_costs = self.arithmetic.costs
        m_costs = self.memory.costs

        q = _intensity(f, m)

        # ...
        self._costs = OrderedDict()
        for i,fi in f_costs.items():
            mi = m_costs[i]
            qi = _intensity(fi, mi)
            self._costs[i] = qi
        # ...

        return q
