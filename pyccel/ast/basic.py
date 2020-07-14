from sympy.core.basic import Basic as sp_Basic

__all__ = ('Basic', 'PyccelAstNode')

#==============================================================================
class Basic(sp_Basic):
    """Basic class for Pyccel AST."""
    _fst = None

    def set_fst(self, fst):
        """Sets the python.ast fst."""
        self._fst = fst

    @property
    def fst(self):
        return self._fst

class PyccelAstNode:
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

