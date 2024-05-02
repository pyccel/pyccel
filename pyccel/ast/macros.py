# coding: utf-8

from sympy.core.expr import AtomicExpr
from sympy import sympify

from .basic          import Basic, PyccelAstNode
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

    def __new__(cls, argument):
        # TODO add verification

        argument = sympify(argument, locals=local_sympify)
        return Basic.__new__(cls, argument)

    @property
    def argument(self):
        return self._args[0]

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

    def __new__(cls, argument, index=None):
        return Macro.__new__(cls, argument)

    def __init__(self, argument, index=None):
        self._index = index

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

    def __new__(cls, argument):
        return Macro.__new__(cls, argument)

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

    def __new__(cls, argument):
        return Macro.__new__(cls, argument)

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

