# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This module provides us with functions and objects that allow us to compute
the arithmetic complexity a of a program.

Example
-------

>>> code = '''
... n = 10
... for i in range(0,n):
...     for j in range(0,n):
...         x = pow(i,2) + pow(i,3) + 3*i
...         y = x / 3 + 2* x
... '''

>>> from pyccel.complexity.memory import MemComplexity
>>> M = OpComplexity(code)
>>> d = M.cost()
>>> print "f = ", d['f']
f =  n**2*(2*ADD + DIV + 2*MUL + 2*POW)

"""

from sympy import count_ops as sympy_count_ops
from sympy import Tuple

from pyccel.ast.core     import For, Assign, CodeBlock, Comment
from pyccel.ast.numpyext import NumpyZeros, NumpyOnes
from pyccel.ast.sympy_helper import pyccel_to_sympy
from pyccel.complexity.basic import Complexity

__all__ = ["count_ops", "OpComplexity"]

class OpComplexity(Complexity):
    """class for Operation complexity computation."""

    def cost(self):
        """
        Computes the complexity of the given code.

        verbose: bool
            talk more
        """
        return count_ops(self.ast, visual=True)


def count_ops(expr, visual=None):

    symbol_map = {}
    used_names = set()

    if isinstance(expr, Assign):
        rhs = pyccel_to_sympy(expr.rhs, symbol_map, used_names)
        return sympy_count_ops(rhs, visual)
    elif isinstance(expr, For):
        a = pyccel_to_sympy(expr.iterable, symbol_map, used_names).size
        ops = sum(count_ops(i, visual) for i in expr.body.body)
        return a*ops
    elif isinstance(expr, CodeBlock):
        return sum(count_ops(i, visual) for i in expr.body)
    elif isinstance(expr, (NumpyZeros, NumpyOnes, Comment)):
        return 0

    expr = pyccel_to_sympy(expr, symbol_map, used_names)

    if isinstance(expr, Tuple):
        return sum(count_ops(i, visual) for i in expr)
    else:
        raise NotImplementedError('TODO count_ops for {}'.format(type(expr)))


##############################################
if __name__ == "__main__":
    code = '''
n = 10

for i in range(0,n):
    for j in range(0,n):
        x = pow(i,2) + pow(i,3) + 3*i
        y = x / 3 + 2* x
    '''

    complexity = OpComplexity(code)
    print((complexity.cost()))
