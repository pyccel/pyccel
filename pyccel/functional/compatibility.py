"""
This file contains functions and classes needed for the functional programing
feature of Pyccel.
"""

__all__ = ['add', 'mul', 'where', 'Where', '_']

import operator

#==============================================================================
class Where(dict):
    pass

#==============================================================================
class AnyArgument(object):
    """a class representing any argument."""
    pass

#==============================================================================
# user friendly
_ = AnyArgument()
where = Where
add = operator.add
mul = operator.add
