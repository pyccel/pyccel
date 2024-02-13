# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
The List container has a number of built-in methods that are 
always available.

This module contains objects which describe these methods within Pyccel's AST.
"""

from pyccel.ast.internals import PyccelInternalFunction
from pyccel.ast.datatypes import NativeHomogeneousList

class ListPop(PyccelInternalFunction) :
    """
    Represents a call to the .pop() method.
    
    Represents a call to the .pop() method which
    removes the item at the specified index. 
    The method also returns the removed item.

    Parameters
    ----------
    list_variable : TypedAstNode
        The name of the list.

    index_element : TypedAstNode
        The current index value for the element to be popped.
    """
    __slots__ = ('_dtype','_precision', '_index','_list_variable')
    _attribute_nodes = ('_index','_list_variable')
    _rank = 0
    _order = None
    _shape = None
    _class_type = NativeHomogeneousList()
    name = 'pop'
    def __init__(self, list_variable, index_element=None):
        self._index = index_element
        self._list_variable = list_variable
        self._dtype = list_variable.dtype
        self._precision = list_variable.precision
        super.__init__()

    @property
    def pop_index(self):
        """
        The current index value for the element to be popped.

        The current index value for the element to be popped.
        """
        return self._index

    @property
    def list_variable(self):
        """
        Provide the name of the list as the return value.
        
        Provide the name of the list as the return value.
        """
        return self._list_variable
