from sympy.core.basic import Basic as sp_Basic

__all__ = ('Basic',)

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
