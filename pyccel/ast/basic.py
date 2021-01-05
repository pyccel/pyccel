#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

"""
This module contains two classes. Basic that provides a python AST and PyccelAstNode which describes each PyccelAstNode
"""

from sympy.core.basic import Basic as sp_Basic

__all__ = ('Basic', 'PyccelAstNode')

#==============================================================================
class Basic(sp_Basic):
    """Basic class for Pyccel AST."""
    _fst = None
    def __init__(self, children):
        self._parent = None
        self._children = children
        for c in children.values():
            c.parent = self

    def has_parent_of_type(self, search_type):
        if isinstance(self._parent, search_type):
            return True
        elif self._parent:
            return self._parent.has_parent_of_type(search_type)
        else:
            return False

    def substitute(self, original, replacement, excluded_nodes = ()):
        """
        Substitute object original for object replacement in the code.
        Any types in excluded_nodes will not be visited

        Parameters
        ==========
        original    : object
                      The original object to be replaced
        replacement : object
                      The object which will be inserted instead
        excluded_nodes : object
                      Types for which substitute should not be called
        """
        for n,v in self._children:
            if v is original:
                setattr(self,n, replacement)
            elif not v.is_atomic and not isinstance(v, excluded_nodes):
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
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent

class PyccelAstNode(Basic):
    stage      = None
    _shape     = None
    _rank      = None
    _dtype     = None
    _precision = None
    _order     = None

    @property
    def shape(self):
        return self._shape

    @property
    def rank(self):
        return self._rank

    @property
    def dtype(self):
        return self._dtype

    @property
    def precision(self):
        return self._precision

    @property
    def order(self):
        return self._order

    def copy_attributes(self, x):
        self._shape     = x.shape
        self._rank      = x.rank
        self._dtype     = x.dtype
        self._precision = x.precision
        self._order     = x.order

