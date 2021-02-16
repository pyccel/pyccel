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
from pyccel.ast.internals    import Slice
from pyccel.ast.literals     import Literal
from pyccel.ast.numpyext     import NumpyZeros, NumpyOnes
from pyccel.ast.sympy_helper import pyccel_to_sympy
from pyccel.complexity.basic import Complexity
from pyccel.complexity.basic import SHAPE


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
        # TODO add other numpy array constructors
        if isinstance(expr.rhs, (NumpyZeros, NumpyOnes)):
            shape = [pyccel_to_sympy(i, self._symbol_map, self._used_names) for i in expr.rhs.shape]
            size = 1
            for i in shape:
                size *= i

            self._shapes[expr.lhs] = shape

            return size * WRITE

        ntimes = self._compute_size_lhs(expr)
        return  self._cost(expr.rhs, **settings) + ntimes * WRITE

    def _cost_AugAssign(self, expr, **settings):
        ntimes = self._compute_size_lhs(expr)
        return  self._cost(expr.rhs, **settings) + ntimes * WRITE

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

    def _cost_PyccelPow(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_PythonAbs(self, expr, **settings):
        return sum(self._cost(i, **settings) for i in expr.args)

    def _cost_Variable(self, expr, **settings):
        return READ

    def _cost_PyccelSymbol(self, expr, **settings):
        return READ

    def _cost_IndexedElement(self, expr, **settings):

        ntimes = 1
        indices = [(e,i) for e,i in enumerate(expr.indices) if isinstance(i, Slice)]
        for e,i in indices:
            # ...
            start = 0
            if not i.start == None:
                if isinstance(i.start, Literal):
                    start = i.start.python_value
                else:
                    start = pyccel_to_sympy(i.start, self._symbol_map, self._used_names)
            # ...

            # ...
            stop = SHAPE(expr.base, e)
            if not i.stop == None:
                if isinstance(i.stop, Literal):
                    stop = i.stop.python_value
                else:
                    stop = pyccel_to_sympy(i.stop, self._symbol_map, self._used_names)
            # ...

            # ...
            step = 1
            if not i.step == None:
                if isinstance(i.step, Literal):
                    step = i.step.python_value
                else:
                    step = pyccel_to_sympy(i.step, self._symbol_map, self._used_names)
            # ...

            if not(step == 1):
                raise NotImplementedError('only step == 1 is treated')

            # TODO uncomment this
            #      this was commented because we get floor(...)
            ntimes *= (stop - start) #// step

        return ntimes * READ

    def _cost_Allocate(self, expr, **settings):
        # TODO
        return 0

    def _cost_Deallocate(self, expr, **settings):
        # TODO
        return 0

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
