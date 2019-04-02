#!/usr/bin/python
# -*- coding: utf-8 -*-

from .basic import Basic
from sympy.core.expr  import Expr, AtomicExpr
from sympy import Tuple

class ListComprehension(Basic):
    """a representation for list comprehension statments."""

    def __new__( cls, iterator, iterable, expr ):
        # ...
        if not isinstance(iterator, (tuple, list, Tuple)):
            raise TypeError('Expecting an iterable')

        iterator = Tuple(*iterator)
        # ...

        # ...
        if not isinstance(iterable, (tuple, list, Tuple)):
            raise TypeError('Expecting an iterable')

        iterable = Tuple(*iterable)
        # ...

        return Basic.__new__(cls, iterator, iterable, expr)

    @property
    def iterator(self):
        return self._args[0]

    @property
    def iterable(self):
        return self._args[1]

    @property
    def expr(self):
        return self._args[2]
