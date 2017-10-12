# coding: utf-8

# TODO - SPL_comm_gatherv: needs a new data structure for the variable

import numpy as np

from pyccel.types.ast import FunctionCall


##########################################################
#
##########################################################
# check arguments
class SPL_EvalBasisFunsDers(FunctionCall):
    _name   = 'EvalBasisFunsDers'
    _module = 'bsp'

    """
    Represents a EvalBasisFunsDers from spl.

    Examples

    >>> from pyccel.clapp.spl import SPL_EvalBasisFunsDers
    >>> from pyccel.types.ast import Variable
    >>> p    = Variable('int', 'p')
    >>> m    = Variable('int', 'm')
    >>> U    = Variable('double', 'U', rank=1, shape=m+1, allocatable=True)
    >>> uu   = Variable('double', 'uu')
    >>> d    = Variable('int', 'd')
    >>> span = Variable('int', 'span')
    >>> dN   = Variable('double', 'dN', rank=2, shape=(p+1,d+1), allocatable=True)
    >>> SPL_EvalBasisFunsDers(p,m,U,uu,d,span,dN)
    EvalBasisFunsDers(p, m, U, uu, d, span, dN)
    """
    def __new__(cls, p, m, U, uu, d, span, dN):
        args = (p, m, U, uu, d, span, dN)
        return super(SPL_EvalBasisFunsDers, cls).__new__(cls, cls._name, args, kind='procedure')

    @property
    def name(self):
        return self._name

    @property
    def module(self):
        return self._module

##########################################################
