from sympy.core.basic import Basic as sp_Basic

__all__ = ('Basic', 'PyccelAstNode')

#==============================================================================
class Basic(sp_Basic):
    """Basic class for Pyccel AST."""
    _fst = None

    def set_fst(self, fst):
        """Sets the redbaron fst."""
        self._fst = fst

    @property
    def fst(self):
        return self._fst

class PyccelAstNode:
    _shape     = None
    _rank      = None
    _dtype     = None
    _precision = None

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
