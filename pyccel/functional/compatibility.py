"""
This file contains functions and classes needed for the functional programing
feature of Pyccel.
"""

__all__ = ['xmap', 'tmap', 'xproduct', 'add', 'mul', '_']

import operator

#==============================================================================
class AnyArgument(object):
    """a class representing any argument."""
    pass

#==============================================================================
# user friendly
_ = AnyArgument()
add = operator.add
mul = operator.add

#==============================================================================
from numpy import reshape
from itertools import product

# useful function used in this file
_len = lambda *args: [len(i) for i in args]

xproduct = lambda    *args: zip(*product(*args))
xmap     = lambda f, *args: map(f, *xproduct(*args))
tmap     = lambda f, *args: reshape(list(xmap(f, *args)), _len(*args))
#==============================================================================
