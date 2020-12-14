#--------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details. #
#--------------------------------------------------------------------------------------#
"""
This module represent a call to the itertools functions for code generation.
"""
from sympy import Tuple
from .basic     import Basic

__all__ = (
    'Product',
)

class Product(Basic):
    """
    Represents a call to itertools.product for code generation.

    arg : list ,tuple ,Tuple
    """

    def __new__(cls, *args):
        if not isinstance(args, (tuple, list, Tuple)):
            raise TypeError('args must be an iterable')
        elif len(args) < 2:
            return args[0]
        return Basic.__new__(cls, *args)

    @property
    def elements(self):
        """get expression's elements"""
        return self._args
