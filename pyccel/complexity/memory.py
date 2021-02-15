# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This module provides us with functions and objects that allow us to compute
the memory complexity a of a program.

Example
-------
"""

from sympy import sympify, Symbol
from sympy import Poly, LT
from sympy import summation

from pyccel.ast.basic        import Basic
from pyccel.ast.builtins     import PythonTuple
from pyccel.ast.core         import For, Assign, CodeBlock
from pyccel.ast.internals    import PyccelSymbol
from pyccel.ast.numpyext     import NumpyZeros, NumpyOnes
from pyccel.ast.sympy_helper import pyccel_to_sympy
from pyccel.complexity.basic import Complexity
from pyccel.complexity.arithmetic import _compute_size_lhs


__all__ = ["count_access", "MemComplexity"]

WRITE = Symbol('WRITE')
READ  = Symbol('READ')


# ...
def count_access(expr, visual=True):
    """
    returns the number of access to memory in terms of WRITE and READ.

    expr: sympy.Expr
        any sympy expression or pyccel.ast.core object
    visual: bool
        If ``visual`` is ``True`` then the number of each type of operation is shown
        with the core class types (or their virtual equivalent) multiplied by the
        number of times they occur.
    local_vars: list
        list of variables that are supposed to be in the fast memory. We will
        ignore their corresponding memory accesses.
    """


    symbol_map = {}
    used_names = set()

    if isinstance(expr, Assign):
        return count_access(expr.rhs, visual) + WRITE

    elif isinstance(expr, (NumpyZeros, NumpyOnes)):
        import numpy as np
        return WRITE*np.prod(expr.shape)

    elif isinstance(expr, Basic):

        atoms = expr.get_attribute_nodes(PyccelSymbol)
        return READ*len(atoms)

    else:
        raise NotImplementedError('TODO count_access for {}'.format(type(expr)))

# ...
class MemComplexity(Complexity):
    """
    Class for memory complexity computation.
    This class implements a simple two level memory model

    Example

    """

    def _cost_Assign(self, expr, **settings):
        ntimes = _compute_size_lhs(expr)
        return ntimes * ( self._cost(expr.rhs, **settings) + WRITE )

    def _cost_AugAssign(self, expr, **settings):
        ntimes = _compute_size_lhs(expr)
        return ntimes * ( self._cost(expr.rhs, **settings) + WRITE )

    def _cost_PyccelAdd(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_PyccelMinus(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_PyccelDiv(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_PyccelFloorDiv(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_PyccelMul(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_PythonAbs(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_Variable(self, expr, **settings):
        return READ

    def _cost_PyccelSymbol(self, expr, **settings):
        return READ

    def _cost_NumpyFloor(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpyExp(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpyLog(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpySqrt(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpySin(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpyCos(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpyTan(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpyArcsin(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpyArccos(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpyArctan(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpyArctan2(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpySinh(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpyCosh(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpyTanh(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpyArcsinh(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpyArccosh(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_NumpyArctanh(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)
