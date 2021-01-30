#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This module contains classes from which all pyccel nodes inherit.
They are:
- Basic, which provides a python AST
- PyccelAstNode which describes each PyccelAstNode
"""

from sympy.core.basic import Basic as sp_Basic

__all__ = ('Basic', 'PyccelAstNode')

#==============================================================================
class Basic(sp_Basic):
    """Basic class for Pyccel AST."""
    _fst = None

    def __new__(cls, *args, **kwargs):
        hashable_args  = [a if not isinstance(a, list) else tuple(a) for a in args]
        return sp_Basic.__new__(cls, *hashable_args)

    def __init__(self):
        self._parent = []
        for c_name in self._children:
            c = getattr(self, c_name)
            if isinstance(c, tuple):
                for ci in c:
                    if isinstance(ci, tuple): # TODO: Fix if to avoid multi-layers
                        for cii in ci:
                            cii.parent = self
                    else:
                        ci.parent = self
            elif hasattr(c, 'parent'): # TODO: Necessary while Symbol is sympy
                c.parent = self # TODO: write set_parent. Handled differently for e.g. Variable which exists in multiple places

    def has_parent_of_type(self, search_type):
        """ Find out if any of the parents are instances
        of the provided class

        Parameters
        ----------
        search_type : ClassType or tuple of ClassTypes
                      The types which we are looking for

        Results
        -------
        Boolean : True if one of the parents is an instance of
                  the class in the argument
        """
        if len(self._parent) == 0:
            return False
        else:
            return any(isinstance(p, search_type) or \
                    p.has_parent_of_type(search_type) \
                    for p in self._parent)

    def contains_type(self, search_type):
        """ Find out if there is an object of the requested type
        in the current object

        Parameters
        ----------
        search_type : ClassType or tuple of ClassTypes
                      The types which we are looking for

        Results
        -------
        Boolean : True if the object contains an object of the
                  requested type
        """
        for n in self._children:
            v = getattr(self, n)
            if isinstance(v, search_type):
                return True
            elif isinstance(v, tuple):
                if any(isinstance(vi, search_type) or \
                        vi.contains_type(search_type) \
                        for vi in v):
                    return True
            elif v.contains_type(search_type):
                return True
        return False

    def substitute(self, original, replacement, excluded_nodes = ()):
        """
        Substitute object 'original' for object 'replacement' in the code.
        Any types in excluded_nodes will not be visited

        Parameters
        ==========
        original    : object
                      The original object to be replaced
        replacement : object
                      The object which will be inserted instead
        excluded_nodes : tuple of types
                      Types for which substitute should not be called
        """
        for n in self._children:
            v = getattr(self, n)
            if v is original:
                setattr(self, n, replacement)
            elif isinstance(v, tuple):
                if original in v:
                    v = tuple(replacement if vi is original else vi for vi in v)
                for vi in v:
                    v.substitute(original, replacement, excluded_nodes)
                setattr(self, n, v)
            elif not isinstance(v, excluded_nodes):
                v.substitute(original, replacement, excluded_nodes)

    @property
    def is_atomic(self):
        """ Indicates whether the object has any children
        """
        return bool(self._children)

    def set_fst(self, fst):
        """Sets the python.ast fst."""
        self._fst = fst

    @property
    def fst(self):
        return self._fst

    @property
    def parent(self):
        """ Get the current parent of the object
        """
        if self._parent:
            return self._parent[-1]
        else:
            return None

    @parent.setter
    def parent(self, parent):
        self._parent.append(parent)

class PyccelAstNode(Basic):
    """Class from which all nodes containing objects inherit
    """
    stage      = None
    _shape     = None
    _rank      = None
    _dtype     = None
    _precision = None
    _order     = None

    @property
    def shape(self):
        """ Tuple containing the length of each dimension
        of the object """
        return self._shape

    @property
    def rank(self):
        """ Number of dimensions of the object
        """
        return self._rank

    @property
    def dtype(self):
        """ Datatype of the object """
        return self._dtype

    @property
    def precision(self):
        """ Precision of the datatype of the object """
        return self._precision

    @property
    def order(self):
        """ Indicates whether the data is stored in
        row-major ('C') or column-major ('F') format.
        This is only relevant if rank > 1 """
        return self._order

    def copy_attributes(self, x):
        self._shape     = x.shape
        self._rank      = x.rank
        self._dtype     = x.dtype
        self._precision = x.precision
        self._order     = x.order

