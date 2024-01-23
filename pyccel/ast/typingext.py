#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
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
    """
    __slots__ = ()

#==============================================================================

typing_funcs = {
        'Final': PyccelFunctionDef('Final', TypingFinal),
    }

typing_mod = Module('typing',
    variables = (),
    funcs = typing_funcs.values(),
    )
