# coding: utf-8

import numpy as np

from pyccel.ast.core import FunctionDef
from pyccel.ast.core import Variable


##########################################################
#
##########################################################
# check arguments
class SPL_EvalBasisFunsDers(FunctionDef):
    _module = 'bsp'

    """
    Represents a EvalBasisFunsDers from spl.

    Examples

    >>> from pyccel.clapp.spl import SPL_EvalBasisFunsDers
    >>> SPL_EvalBasisFunsDers()
    span, dN := EvalBasisFunsDers(p, m, U, uu, d)
    """
    def __new__(cls):
        # ...
        p    = Variable('int', 'p')
        m    = Variable('int', 'm')
        U    = Variable('double', 'U', \
                        rank=1, shape=m+1, \
                        allocatable=True)
        uu   = Variable('double', 'uu')
        d    = Variable('int', 'd')
        span = Variable('int', 'span')
        dN   = Variable('double', 'dN', \
                        rank=2, shape=(p+1,d+1), \
                        allocatable=True)
        # ...

        # ...
        name        = 'EvalBasisFunsDers'
        body        = []
        local_vars  = []
        global_vars = []
        hide        = True
        kind        = 'procedure'
        args        = [p, m, U, uu, d]
        results     = [span, dN]
        # ...

        return FunctionDef.__new__(cls, name, args, results, \
                                   body, local_vars, global_vars, \
                                   hide=hide, kind=kind)

    @property
    def module(self):
        return self._module

    def _sympystr(self, printer):
        sstr = printer.doprint

        name    = sstr(self.name)
        args    = ', '.join(sstr(i) for i in self.arguments)
        results = ', '.join(sstr(i) for i in self.results)

        code = '{2} := {0}({1})'.format(name, args, results)
        return code

##########################################################
