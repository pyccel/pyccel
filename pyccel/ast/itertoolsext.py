#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
This module represent a call to the itertools functions for code generation.
"""
from .core      import PyccelFunctionDef, Module
from .internals import Iterable

__all__ = (
    'Product',
    'itertools_mod',
)

class Product(Iterable):
    """
    Represents a call to itertools.product for code generation.

    Represents a call to itertools.product for code generation.

    Parameters
    ----------
    *args : PyccelAstType
        The arguments passed to the product function.
    """
    __slots__ = ('_elements',)
    _attribute_nodes = ('_elements',)

    def __new__(cls, *args):
        if not isinstance(args, (tuple, list)):
            raise TypeError('args must be an iterable')
        elif len(args) < 2:
            return args[0]
        else:
            return super().__new__(cls)

    def __init__(self, *args):
        self._elements = args
        super().__init__(len(args))

    @property
    def elements(self):
        """get expression's elements"""
        return self._elements

    def get_python_iterable_item(self):
        """
        Get the item of the iterable that will be saved to the loop targets.

        This is an element from each of the variables indexed using the
        iterators previously provided via the set_loop_counters method.

        Returns
        -------
        list[TypedAstNode]
            A list of objects that should be assigned to variables.
        """
        return [elem[idx] for idx, elem in zip(self._indices, self.elements)]

#==============================================================================
itertools_mod = Module('itertools',(),
        funcs = [PyccelFunctionDef('product',Product)])
