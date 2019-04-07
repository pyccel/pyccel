#!/usr/bin/python
# -*- coding: utf-8 -*-

from sympy import Tuple

from pyccel.ast.basic import Basic
from pyccel.ast.core  import Variable
from pyccel.codegen.utilities import random_string

#==============================================================================
class BasicMap(Basic):
    """."""

    def __new__( cls, func, target ):

        return Basic.__new__(cls, func, target)

    @property
    def func(self):
        return self._args[0]

    @property
    def target(self):
        return self._args[1]

class BasicTensorMap(BasicMap):
    pass

#==============================================================================
class Reduce(Basic):
    """."""

    def __new__( cls, func, target ):

        return Basic.__new__(cls, func, target)

    @property
    def func(self):
        return self._args[0]

    @property
    def target(self):
        return self._args[1]

#==============================================================================
class BasicGenerator(Basic):
    def __new__( cls, *args ):
        return Basic.__new__(cls, args)

    @property
    def arguments(self):
        return self._args[0]

    def __len__(self):
        return len(self.arguments)


#==============================================================================
# serial and parallel nodes
class SeqMap(BasicMap):
    pass

class ParMap(BasicMap):
    pass

class SeqTensorMap(BasicTensorMap):
    pass

class ParTensorMap(BasicTensorMap):
    pass

class SeqZip(BasicGenerator):
    pass

class ParZip(BasicGenerator):
    pass

class SeqProduct(BasicGenerator):
    pass

class ParProduct(BasicGenerator):
    pass

#==============================================================================
class BasicTypeVariable(Basic):
    pass

#==============================================================================
class TypeVariable(BasicTypeVariable):
    _name = None
    def __new__( cls, var, rank=0 ):
        assert(isinstance(var, (Variable, TypeVariable)))

        dtype          = var.dtype
        rank           = var.rank + rank
        is_stack_array = var.is_stack_array
        order          = var.order
        precision      = var.precision

        obj = Basic.__new__(cls, dtype, rank, is_stack_array, order, precision)

        obj._name = 'tv_{}'.format( random_string( 4 ) )

        return obj

    @property
    def dtype(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

    @property
    def is_stack_array(self):
        return self._args[2]

    @property
    def order(self):
        return self._args[3]

    @property
    def precision(self):
        return self._args[4]

    @property
    def name(self):
        return self._name

    def incr_rank(self, value):
        return TypeVariable( self, rank=value+self.rank )

    def _sympystr(self, printer):
        sstr = printer.doprint
        return sstr(self.name)

    def view(self):
        """inspects the variable."""
        attributs = self._args[:]
        attributs = ','.join(str(i) for i in attributs)
        return 'TypeVariable({})'.format(attributs)

#==============================================================================
class TypeTuple(BasicTypeVariable):
    _name = None
    def __new__( cls, var, rank=0 ):
        assert(isinstance(var, (tuple, list, Tuple)))

        for i in var:
            assert( isinstance(i, (Variable, TypeVariable)) )

        t_vars = []
        for i in var:
            t_var = TypeVariable( i, rank=rank )
            t_vars.append(t_var)

        t_vars = Tuple(*t_vars)

        obj = Basic.__new__(cls, t_vars)

        obj._name = 'tt_{}'.format( random_string( 4 ) )

        return obj

    @property
    def types(self):
        return self._args[0]

    @property
    def name(self):
        return self._name

    def _sympystr(self, printer):
        sstr = printer.doprint
        return sstr(self.name)

    def view(self):
        """inspects the variable."""
        attributs = ','.join(i.view() for i in self.types)
        return 'TypeTuple({})'.format(attributs)

#==============================================================================
class TypeList(BasicTypeVariable):
    _name = None
    def __new__( cls, var ):
        assert(isinstance(var, (TypeVariable, TypeTuple)))

        obj = Basic.__new__(cls, var)

        obj._name = 'tl_{}'.format( random_string( 4 ) )

        return obj

    @property
    def parent(self):
        return self._args[0]

    @property
    def name(self):
        return self._name

    def _sympystr(self, printer):
        sstr = printer.doprint
        return sstr(self.name)

    def view(self):
        """inspects the variable."""
        return 'TypeList({})'.format(self.parent.view())

#==============================================================================
# user friendly function
def assign_type(expr, rank=None):
    if ( rank is None ) and isinstance(expr, BasicTypeVariable):
        return expr

    if rank is None:
        rank = 0

    if isinstance(expr, (Variable, TypeVariable)):
        return TypeVariable(expr, rank=rank)

    elif isinstance(expr, (tuple, list, Tuple)):
        if len(expr) == 1:
            return assign_type(expr[0], rank=rank)

        else:
            return TypeTuple(expr)

    elif isinstance(expr, TypeTuple):
#        ls = [assign_type( i, rank=rank ) for i in expr.types]
#        return assign_type( ls )
        raise NotImplementedError('')

#    elif isinstance(expr, TypeTuple):
#        return TypeTuple(expr, rank=rank)

    else:
        raise TypeError('> wrong argument, given {}'.format(type(expr)))
