
from sympy import Tuple
from .basic     import Basic

__all__ = (
    'Product',
)

class Product(Basic):
    """Represents a Product stmt."""

    def __new__(cls, *args):
        if not isinstance(args, (tuple, list, Tuple)):
            raise TypeError('args must be an iterable')
        elif len(args) < 2:
            return args[0]
        return Basic.__new__(cls, *args)

    @property
    def elements(self):
        return self._args
