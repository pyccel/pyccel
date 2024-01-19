#!/usr/bin/python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
""" Module containing objects from the typing module understood by pyccel
"""

from .core      import Module
from .datatypes import NativeSymbol
from .variable  import Variable

__all__ = (
    'TypingFinal',
    'typing_mod'
)

#==============================================================================

typing_constants = {
        'Final': Variable(NativeSymbol(), 'Final'),
    }

typing_mod = Module('typing',
    variables = typing_constants.values(),
    funcs = (),
    )
