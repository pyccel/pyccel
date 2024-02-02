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
from ..datatypes import (NativeInteger, NativeHomogeneousList, NativeVoid)

from pyccel.ast.basic import PyccelAstNode, TypedAstNode
from pyccel.ast.internals import PyccelInternalFunction

pyccel_stage = PyccelStage()


class ListPop(PyccelInternalFunction) :
    """
    Represents a call to the .pop() method.
    
    Represents a call to the .pop() method wich 
    removes the item at the specified index. 
    The method also returns the removed item.
    
    Parameters
    ----------
    arg : TypedAstNode
       
    """
    __slots__ = ('_dtype')
    _precision = -1
    _rank  = 0
    _shape = None
    _order = None
    _class_type = NativeHomogeneousList()
    name = 'pop'
    def __init__(self, *args):
        super().__init__(*args)
        self._dtype = args[0].dtype
        