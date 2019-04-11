#!/usr/bin/python
# -*- coding: utf-8 -*-

from sympy import Tuple, IndexedBase

from pyccel.ast.basic import Basic
from pyccel.ast.core  import Variable, For, Range, Assign, Len
from pyccel.codegen.utilities import random_string


#=========================================================================
class BasicTypeVariable(Basic):
    _tag  = None

    @property
    def tag(self):
        return self._tag

#=========================================================================
class TypeVariable(BasicTypeVariable):
    def __new__( cls, var, rank=0 ):
        assert(isinstance(var, (Variable, TypeVariable)))

        dtype          = var.dtype
        rank           = var.rank + rank
        is_stack_array = var.is_stack_array
        order          = var.order
        precision      = var.precision
        shape          = var.shape

        obj = Basic.__new__( cls, dtype, rank, is_stack_array, order, precision, shape )
        obj._tag = random_string( 4 )

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
    def shape(self):
        return self._args[5]

    @property
    def name(self):
        return 'tv_{}'.format(self.tag)

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

    def duplicate(self):
        return TypeVariable( self )

#=========================================================================
class TypeTuple(BasicTypeVariable):
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
        obj._tag = random_string( 4 )

        return obj

    @property
    def types(self):
        return self._args[0]

    @property
    def name(self):
        return 'tt_{}'.format(self.tag)

    def __len__(self):
        return len(self.types)

    def _sympystr(self, printer):
        sstr = printer.doprint
        types = ','.join(sstr(i) for i in self.types)
        return '({})'.format(types)

    def view(self):
        """inspects the variable."""
        attributs = ','.join(i.view() for i in self.types)
        return 'TypeTuple({})'.format(attributs)

    def duplicate(self):
        raise NotImplementedError('')

#=========================================================================
class TypeList(BasicTypeVariable):
    def __new__( cls, var ):
        assert(isinstance(var, (TypeVariable, TypeTuple, TypeList)))

        obj = Basic.__new__(cls, var)
        obj._tag = random_string( 4 )

        # ...
        def _get_core_type(expr):
            if isinstance(expr, TypeList):
                return _get_core_type(expr.parent)

            else:
                return expr
        # ...

        obj._types = _get_core_type(var)

        return obj

    @property
    def parent(self):
        return self._args[0]

    @property
    def name(self):
        return 'tl_{}'.format(self.tag)

    @property
    def types(self):
        return self._types

    def __len__(self):
        n = 1
        if isinstance(self.parent, TypeList):
            n += len(self.parent)

        return n

    def _sympystr(self, printer):
        sstr = printer.doprint
        types = sstr(self.types)
        for i in range(0, len(self)):
            types = '[{}]'.format(types)
        return types

    def view(self):
        """inspects the variable."""
        return 'TypeList({})'.format(self.parent.view())

    def duplicate(self):
        raise NotImplementedError('')

#=========================================================================
# user friendly function
# TODO DO WE KEEP IT?
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
        ls = [assign_type( i, rank=rank ) for i in expr.types]
        return assign_type( ls )

    else:
        raise TypeError('> wrong argument, given {}'.format(type(expr)))
