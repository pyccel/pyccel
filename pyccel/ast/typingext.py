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

__all__ = (
    'TypingFinal',
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

typing_funcs = {
        'Final': PyccelFunctionDef('Final', TypingFinal),
    }

typing_mod = Module('typing',
    variables = (),
    funcs = typing_funcs.values(),
    )
