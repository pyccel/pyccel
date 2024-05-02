#!/usr/bin/python
# -*- coding: utf-8 -*-

from .basic import Basic
from sympy.core.expr  import AtomicExpr

__all__ = (
    'FunctionalFor',
    'FunctionalMap',
    'FunctionalMax',
    'FunctionalMin',
    'FunctionalSum',
    'GeneratorComprehension'
)

#==============================================================================
class FunctionalFor(Basic):

    """."""

    def __new__(
        cls,
        loops,
        expr=None,
        lhs=None,
        indices=None,
        index=None,
        ):
        return Basic.__new__(cls, loops, expr, lhs, indices, index)

    @property
    def loops(self):
        return self._args[0]

    @property
    def expr(self):
        return self._args[1]

    @property
    def lhs(self):
        return self._args[2]

    @property
    def indices(self):
        return self._args[3]

    @property
    def index(self):
        return self._args[4]

#==============================================================================
class GeneratorComprehension(AtomicExpr, Basic):
    pass

#==============================================================================
class FunctionalSum(GeneratorComprehension, FunctionalFor):
    name = 'sum'

#==============================================================================
class FunctionalMax(GeneratorComprehension, FunctionalFor):
    name = 'max'
#==============================================================================

class FunctionalMin(GeneratorComprehension, FunctionalFor):
    name = 'min'

#==============================================================================
class FunctionalMap(GeneratorComprehension, FunctionalFor):
    pass
