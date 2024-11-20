#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
"""
This module represent a call to the itertools functions for code generation.
"""
from .builtins  import PythonLen, PythonRange
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

    def get_target_from_range(self):
        return [elem[idx] for idx, elem in zip(self._indices, self.elements)]

    def get_range(self):
        """ Get the range iterator(s) needed to access all elements of Product
        """
        lengths = [getattr(e, '__len__',
                getattr(e, 'length', PythonLen(e))) for e in self.elements]
        lengths = [l() if callable(l) else l for l in lengths]
        return [PythonRange(l) for l in lengths]

    @property
    def n_indices(self):
        """ Number of indices required
        """
        return len(self._elements)

#==============================================================================
itertools_mod = Module('itertools',(),
        funcs = [PyccelFunctionDef('product',Product)])
