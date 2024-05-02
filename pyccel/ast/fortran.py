# coding: utf-8

from sympy.core.function import Function
from sympy.core.basic import Basic

__all__ = (
    'Ceil',
    'Dot',
    'Max',
    'Min',
    'Mod',
    'Sign'
)

#==============================================================================
# TODO: - implement all the following objects
class Ceil(Function):
    pass


# TODO: add example
class Min(Function):
    """Represents a 'min' expression in the code."""
    def __new__(cls, *args):
        return Basic.__new__(cls, *args)

# TODO: add example
class Max(Function):
    """Represents a 'max' expression in the code."""
    def __new__(cls, *args):
        return Basic.__new__(cls, *args)

# TODO: add example
class Mod(Function):
    """Represents a 'mod' expression in the code."""
    def __new__(cls, *args):
        return Basic.__new__(cls, *args)

# TODO: improve with __new__ from Function and add example
class Dot(Function):
    """
    Represents a 'dot' expression in the code.

    expr_l: variable
        first variable
    expr_r: variable
        second variable
    """
    def __new__(cls, expr_l, expr_r):
        return Basic.__new__(cls, expr_l, expr_r)

    @property
    def expr_l(self):
        return self.args[0]

    @property
    def expr_r(self):
        return self.args[1]

# TODO: treat as a Function
# TODO: add example
class Sign(Basic):

    def __new__(cls,expr):
        return Basic.__new__(cls, expr)

    @property
    def rhs(self):
        return self.args[0]


