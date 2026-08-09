#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------- #
# This file is part of Pyccel which is released under MIT License. See the  #
# LICENSE file or go to https://github.com/pyccel/pyccel/blob/devel/LICENSE #
# for full license details.                                                 #
# ------------------------------------------------------------------------- #
"""Module containing objects representing the decorators understood by pyccel"""

from .basic import TypedAstNode
from .core import PyccelFunctionDef
from .internals import PyccelFunction
import pyccel.decorators as pyccel_decorators

__all__ = (
    "PythonProperty",
    "pyccel_decorator_funcs",
)

# ==============================================================================


class PythonProperty(TypedAstNode):
    """
    Class representing a call to the property decorator.

    Class representing a call to the property decorator. This object
    will never be constructed. It exists to recognise the use.
    """

    __slots__ = ()
    _attribute_nodes = ()
    name = "property"


# ==============================================================================

pyccel_decorator_funcs = {
    d: PyccelFunctionDef(d, PyccelFunction) for d in pyccel_decorators.__all__
}

property_decorator_func = PyccelFunctionDef("property", PythonProperty)
