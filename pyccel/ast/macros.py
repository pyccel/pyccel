# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This module contains all classes and functions used for handling macros.
"""

from sympy.core.expr import AtomicExpr
from sympy import sympify

from .basic          import PyccelAstNode
from .core           import local_sympify
from .datatypes      import default_precision
from .datatypes      import NativeInteger, NativeGeneric

__all__ = (
    'Macro',
    'MacroCount',
    'MacroShape',
    'MacroType',
    'construct_macro'
)

#==============================================================================
class Macro(AtomicExpr, PyccelAstNode):
    """."""
    _name = '__UNDEFINED__'
    _attribute_nodes = ()

    def __init__(self, argument):
        # TODO add verification

        self._argument = argument
        super().__init__()

    @property
    def argument(self):
        return self._argument

    @property
    def name(self):
        return self._name

#==============================================================================
class MacroShape(Macro):
    """."""
    _name      = 'shape'
    _rank      = 1
    _shape     = ()
    _dtype     = NativeInteger()
    _precision = default_precision['integer']

    def __init__(self, argument, index=None):
        self._index = index
        super().__init__(argument)

    @property
    def index(self):
        return self._index

    def _sympystr(self, printer):
        sstr = printer.doprint
        if self.index is None:
            return 'MacroShape({})'.format(sstr(self.argument))
        else:
            return 'MacroShape({}, {})'.format(sstr(self.argument),
                                               sstr(self.index))

#==============================================================================
class MacroType(Macro):
    """."""
    _name      = 'dtype'
    _dtype     = NativeGeneric()
    _rank      = 0
    _shape     = ()
    _precision = 0

    def _sympystr(self, printer):
        sstr = printer.doprint
        return 'MacroType({})'.format(sstr(self.argument))

#==============================================================================
class MacroCount(Macro):
    """."""
    _name      = 'count'
    _rank      = 0
    _shape     = ()
    _dtype     = NativeInteger()
    _precision = default_precision['integer']

    def _sympystr(self, printer):
        sstr = printer.doprint
        return 'MacroCount({})'.format(sstr(self.argument))





def construct_macro(name, argument, parameter=None):
    """."""
    # TODO add available macros: shape, len, dtype
    if not isinstance(name, str):
        raise TypeError('name must be of type str')

    argument = sympify(argument, locals=local_sympify)
    if name == 'shape':
        return MacroShape(argument, index=parameter)
    elif name == 'dtype':
        return MacroType(argument)
    elif name == 'count':
        return MacroCount(argument)

