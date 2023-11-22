# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This module contains all classes and functions used for handling macros.
"""
from pyccel.utilities.stage import PyccelStage

from .basic          import TypedAstNode
from .datatypes      import NativeInteger, NativeGeneric
from .internals      import PyccelSymbol
from .variable       import Variable

pyccel_stage = PyccelStage()

__all__ = (
    'Macro',
    'MacroCount',
    'MacroShape',
    'MacroType',
    'construct_macro'
)

#==============================================================================
class Macro(TypedAstNode):
    """
    Class representing a macro, ie an inline definition.

    Class representing a macro, ie an inline definition. TO BE DEPRECATED.

    Parameters
    ----------
    argument : PyccelSymbol | Variable
        The argument passed to the macro.
    """
    __slots__ = ('_argument',)
    _name = '__UNDEFINED__'
    _attribute_nodes = ()

    def __init__(self, argument):
        if not isinstance(argument, (PyccelSymbol, Variable)):
            raise TypeError(f"Argument must be a Pyccelsymbol or a Variable not {type(argument)}")

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
    """
    A macro indicating the shape of a variable.

    A macro indicating the shape of a variable. TO BE DEPRECATED.

    Parameters
    ----------
    argument : PyccelSymbol | Variable
        The variable whose shape we are accessing.
    index : int | LiteralInteger
        The index of the element of the shape we are accessing.
    """
    __slots__ = ('_index','_rank','_shape')
    _name      = 'shape'
    _dtype     = NativeInteger()
    _precision = -1
    _order     = None
    _class_type = NativeInteger()

    def __init__(self, argument, index=None):
        if index is not None:
            self._rank = 0
            self._shape = None
        elif pyccel_stage != "syntactic":
            self._rank      = int(argument.rank>1)
            self._shape     = (argument.rank,)
        else:
            self._rank      = 1
            self._shape     = ()
        self._index = index
        super().__init__(argument)

    @property
    def index(self):
        return self._index

    def __str__(self):
        if self.index is None:
            return f'MacroShape({self.argument})'
        else:
            return f'MacroShape({self.argument}, {self.index})'

#==============================================================================
class MacroType(Macro):
    """
    A macro representing the type of a variable.

    A macro representing the type of a variable. TO BE DEPRECATED.

    Parameters
    ----------
    argument : PyccelSymbol | Variable
        The variable whose datatype we are accessing.
    """
    __slots__ = ()
    _name      = 'dtype'
    _dtype     = NativeGeneric()
    _precision = 0
    _rank      = 0
    _shape     = None
    _order     = None
    _class_type = NativeGeneric()

    def __str__(self):
        return f'MacroType({self.argument})'

#==============================================================================
class MacroCount(Macro):
    """
    A macro representing the total number of elements in a variable.

    A macro representing the total number of elements in a variable.
    TO BE DEPRECATED.

    Parameters
    ----------
    argument : PyccelSymbol | Variable
        The variable whose size we are accessing.
    """
    __slots__ = ()
    _name      = 'count'
    _dtype     = NativeInteger()
    _precision = -1
    _rank      = 0
    _shape     = None
    _order     = None
    _class_type = NativeInteger()

    def __str__(self):
        return f'MacroCount({self.argument})'





def construct_macro(name, argument, parameter=None):
    """."""
    # TODO add available macros: shape, len, dtype
    if not isinstance(name, str):
        raise TypeError('name must be of type str')

    if name == 'shape':
        return MacroShape(argument, index=parameter)
    elif name == 'dtype':
        return MacroType(argument)
    elif name == 'count':
        return MacroCount(argument)

