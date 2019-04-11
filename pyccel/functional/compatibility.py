"""
This file contains functions and classes needed for the functional programing
feature of Pyccel.
"""

__all__ = ['add', 'mul', '_']

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
