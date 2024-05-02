#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
This module represent a call to the itertools functions for code generation.
"""
from .internals import PyccelInternalFunction

__all__ = (
    'Product',
)

class Product(PyccelInternalFunction):
    """
    Represents a call to itertools.product for code generation.

    arg : list, tuple
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
        super().__init__(args)

    @property
    def elements(self):
        """get expression's elements"""
        return self._elements
