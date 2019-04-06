#!/usr/bin/python
# -*- coding: utf-8 -*-

from pyccel.ast.basic import Basic
from sympy.core.expr  import Expr, AtomicExpr
from sympy import Tuple

#==============================================================================
class BasicMap(Basic):
    """."""

    def __new__( cls, func, target ):

        return Basic.__new__(cls, func, target)

    @property
    def func(self):
        return self._args[0]

    @property
    def target(self):
        return self._args[1]

class BasicTensorMap(BasicMap):
    pass

#==============================================================================
class Reduce(Basic):
    """."""

    def __new__( cls, func, target ):

        return Basic.__new__(cls, func, target)

    @property
    def func(self):
        return self._args[0]

    @property
    def target(self):
        return self._args[1]

#==============================================================================
class BasicGenerator(Basic):
    def __new__( cls, *args ):

        return Basic.__new__(cls, args)

    @property
    def arguments(self):
        return self._args[0]


#==============================================================================
# serial and parallel nodes
class SeqMap(BasicMap):
    pass

class ParMap(BasicMap):
    pass

class SeqTensorMap(BasicTensorMap):
    pass

class ParTensorMap(BasicTensorMap):
    pass

class SeqZip(BasicGenerator):
    pass

class ParZip(BasicGenerator):
    pass

class SeqProduct(BasicGenerator):
    pass

class ParProduct(BasicGenerator):
    pass
