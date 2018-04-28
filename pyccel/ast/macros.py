# coding: utf-8

from sympy.core.expr import AtomicExpr
from sympy import sympify

from .core import Basic

class Macro(AtomicExpr):
    """."""
    _name = '__UNDEFINED__'

    def __new__(cls, argument):
        # TODO add verification

        argument = sympify(argument)
        return Basic.__new__(cls, argument)

    @property
    def argument(self):
        return self._args[0]

    @property
    def name(self):
        return self._name


class MacroShape(Macro):
    """."""
    _name = 'shape'


def construct_macro(name, argument):
    """."""
    # TODO add available macros: shape, len, dtype
    if not isinstance(name, str):
        raise TypeError('name must be of type str')

    argument = sympify(argument)
    if name == 'shape':
        return MacroShape(argument)
