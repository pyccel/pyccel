#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

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
    _children = ('_loops','_expr', '_lhs', '_indices', '_index')

    def __init__(
        self,
        loops,
        expr=None,
        lhs=None,
        indices=None,
        index=None,
        ):
        self._loops   = loops
        self._expr    = expr
        self._lhs     = lhs
        self._indices = indices
        self._index   = index
        super().__init__()

    @property
    def loops(self):
        return self._loops

    @property
    def expr(self):
        return self._expr

    @property
    def lhs(self):
        return self._lhs

    @property
    def indices(self):
        return self._indices

    @property
    def index(self):
        return self._index

#==============================================================================
class GeneratorComprehension(AtomicExpr, Basic):
    _children = ()

#==============================================================================
class FunctionalSum(GeneratorComprehension, FunctionalFor):
    _children = FunctionalFor._children
    name = 'sum'

#==============================================================================
class FunctionalMax(GeneratorComprehension, FunctionalFor):
    _children = FunctionalFor._children
    name = 'max'
#==============================================================================

class FunctionalMin(GeneratorComprehension, FunctionalFor):
    _children = FunctionalFor._children
    name = 'min'

#==============================================================================
class FunctionalMap(GeneratorComprehension, FunctionalFor):
    _children = FunctionalFor._children
