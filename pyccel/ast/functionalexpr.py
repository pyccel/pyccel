#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from .basic import PyccelAstNode

__all__ = (
    'FunctionalFor',
    'FunctionalMax',
    'FunctionalMin',
    'FunctionalSum',
    'GeneratorComprehension'
)

#==============================================================================
class FunctionalFor(PyccelAstNode):

    """."""
    __slots__ = ('_loops','_expr', '_lhs', '_indices', '_index',
            '_dtype','_precision','_rank','_shape','_order')
    _attribute_nodes = ('_loops','_expr', '_lhs', '_indices', '_index')

    def __init__(
        self,
        loops,
        expr=None,
        lhs=None,
        indices=None,
        index=None,
        ):
        self._loops    = loops
        self._expr    = expr
        self._lhs     = lhs
        self._indices = indices
        self._index   = index
        super().__init__()

        if PyccelAstNode.stage != 'syntactic':
            self._dtype     = lhs.dtype
            self._precision = lhs.precision
            self._rank      = lhs.rank
            self._shape     = lhs.shape
            self._order     = lhs.order

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
class GeneratorComprehension(FunctionalFor):
    """ Super class for all functions which reduce generator expressions to scalars
    """
    __slots__ = ()

    def __repr__(self):
        return '{}([{}]  {}'.format(self.name, self.loops, self.indices)

#==============================================================================
class FunctionalSum(GeneratorComprehension):
    """ Represents a call to sum for a list argument
    >>> sum([i in range(5)])
    """
    __slots__ = ()
    name = 'sum'

#==============================================================================
class FunctionalMax(GeneratorComprehension):
    """ Represents a call to max for a list argument
    >>> max([i in range(5)])
    """
    __slots__ = ()
    name = 'max'
#==============================================================================

class FunctionalMin(GeneratorComprehension):
    """ Represents a call to min for a list argument
    >>> min([i in range(5)])
    """
    __slots__ = ()
    name = 'min'
