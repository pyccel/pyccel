# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from pyccel.ast.internals import PyccelInternalFunction
from pyccel.ast.datatypes import NativeHomogeneousList

class ListPop(PyccelInternalFunction) :
    """
    Represents a call to the .pop() method.
    
    Represents a call to the .pop() method wich 
    removes the item at the specified index. 
    The method also returns the removed item.
    
    Parameters
    ----------
    *args : iterable
        The arguments passed to the function call.
    """
    __slots__ = ('_dtype','_precision','_index','_name')
    _rank = 0
    _order = None
    _shape = None
    _class_type = NativeHomogeneousList()
    name = 'pop'
    def __init__(self, name, index_elemnt=None):
        if index_elemnt:
            super().__init__(name, index_elemnt)
        else:
            super().__init__(name)
        self._index = index_elemnt
        self._name = name
        self._dtype = name.dtype
        self._precision = name.precision

    @property
    def args(self):
        """
        Arguments of the Pop method

        index of element to pop
        """
        return self._index
    @property
    def name(self):
        """
        Name of the object contain the list
        
        """
        return self._name
        