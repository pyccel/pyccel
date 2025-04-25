#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing objects from the typing module understood by pyccel
"""

from .basic     import TypedAstNode
from .core      import Module, PyccelFunctionDef
from .datatypes import TypeAlias

__all__ = (
    'TypingFinal',
    'TypingTypeAlias',
    'typing_mod'
)

#==============================================================================

class TypingFinal(TypedAstNode):
    """
    Class representing a call to the typing.Final construct.

    Class representing a call to the typing.Final construct. A "call" to this
    object looks like an IndexedElement. This is because types are involved.

    Parameters
    ----------
    arg : SyntacticTypeAnnotation
        The annotation which is coerced to be constant.
    """
    __slots__ = ('_arg',)
    _attribute_nodes = ('_arg',)

    def __init__(self, arg):
        self._arg = arg
        super().__init__()

    @property
    def arg(self):
        """
        Get the argument describing the type annotation for an object.

        Get the argument describing the type annotation for an object.
        """
        return self._arg

#==============================================================================
class TypingTypeAlias(TypedAstNode):
    """
    Class representing a call to the typing.TypeAlias construct.

    Class representing a call to the typing.TypeAlias construct. This object
    is only used for type annotations. It is useful for creating a PyccelFunctionDef
    but instances should not be created.
    """
    __slots__ = ()
    _attribute_nodes = ()
    _static_type = TypeAlias()

#==============================================================================
class TypingTypeVar(TypedAstNode):
    """
    Class representing a call to the typing.TypeVar construct.

    Class representing a call to the typing.TypeVar construct. This object
    is a type annotation.
    """
    __slots__ = ('_name', '_possible_types')
    _attribute_nodes = ()
    _class_type = TypeAlias()
    _shape = None

    def __init__(self, name, *constraints, bound=None, covariant=False, contravariant=False):
        if covariant or contravariant:
            raise TypeError("Covariant and contravariant types are not currently supported")
        if len(constraints) == 0:
            raise TypeError(f"The possible types for {name} must be specified")
        self._name = name
        self._possible_types = constraints
        super().__init__()

    @property
    def type_list(self):
        return self._possible_types

#==============================================================================

typing_funcs = {
        'Final': PyccelFunctionDef('Final', TypingFinal),
        'TypeAlias': PyccelFunctionDef('TypeAlias', TypingTypeAlias),
        'TypeVar' : PyccelFunctionDef('TypeVar', TypingTypeVar),
    }

typing_mod = Module('typing',
    variables = (),
    funcs = typing_funcs.values(),
    )
