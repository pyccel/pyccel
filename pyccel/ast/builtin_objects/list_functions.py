# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
The List container has a number of built-in methods that are 
always available.

In this module we implement List methods.
"""

from pyccel.errors.errors import PyccelError

from pyccel.utilities.stage import PyccelStage

from pyccel.ast.basic import PyccelAstNode, TypedAstNode
from pyccel.ast.datatypes import NativeHomogeneousList

pyccel_stage = PyccelStage()

class ListAppend():
    pass

class ListPop():
    pass

class ListInsert():
    pass

class ListClear():
    pass

class ListExtend():
    pass

class ListReverse():
    pass

