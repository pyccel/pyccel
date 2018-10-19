#!/usr/bin/python
# -*- coding: utf-8 -*-

from .basic import Basic

class FunctionalFor(Basic):

    """."""

    def __new__(
        cls,
        loops,
        target,
        indexes,
        index=None,
        ):
        return Basic.__new__(cls, loops, target, indexes, index)

    @property
    def loops(self):
        return self._args[0]

    @property
    def target(self):
        return self._args[1]

    @property
    def indexes(self):
        return self._args[2]

    @property
    def index(self):
        return self._args[3]


class GeneratorComprehension(Basic):

    pass


class FunctionalSum(FunctionalFor, GeneratorComprehension):

    name = 'sum'


class FunctionalMax(FunctionalFor, GeneratorComprehension):

    name = 'max'


class FunctionalMin(FunctionalFor, GeneratorComprehension):

    name = 'min'


class FunctionalMap(FunctionalFor, GeneratorComprehension):

    pass

