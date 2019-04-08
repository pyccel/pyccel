#!/usr/bin/python
# -*- coding: utf-8 -*-

from sympy import Tuple, IndexedBase

from pyccel.ast.basic import Basic
from pyccel.ast.core  import Variable, For, Range, Assign, Len
from pyccel.codegen.utilities import random_string

#=========================================================================
# some useful functions
# TODO add another argument to distinguish between len and other ints
def new_variable( dtype, var ):

    # ...
    if dtype == 'int':
        prefix = 'i'

    elif dtype == 'real':
        prefix = 'x'

    elif dtype == 'len': # for length
        prefix = 'n'
        dtype  = 'int'

    else:
        raise NotImplementedError()
    # ...

    # ...
    tag = random_string( 4 )
    # ...

    pattern = '{prefix}{dim}_{tag}'
    _print = lambda d,t: pattern.format(prefix=prefix, dim=d, tag=t)

    if isinstance( var, Variable ):
        assert( var.rank > 0 )

        if var.rank == 1:
            name  =  _print('', tag)
            return Variable( dtype, name )

        else:
            indices = []
            for d in range(0, var.rank):
                name  =  _print(d, tag)
                indices.append( Variable( dtype, name ) )

            return Tuple(*indices)

    else:
        raise NotImplementedError('{} not available'.format(type(var)))

#=========================================================================
class BasicBlock(Basic):
    """."""
    def __new__( cls, decs, body ):
        assert(isinstance(decs, (tuple, list, Tuple)))
        assert(isinstance(body, (tuple, list, Tuple)))

        decs = Tuple(*decs)
        body = Tuple(*body)

        return Basic.__new__(cls, decs, body)

    @property
    def decs(self):
        return self._args[0]

    @property
    def body(self):
        return self._args[1]

class SequentialBlock(BasicBlock):
    pass

class ParallelBlock(BasicBlock):
    pass

#=========================================================================
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

#=========================================================================
class BasicGenerator(Basic):
    def __new__( cls, *args ):
        # ... create iterator and index variables
        target = args
        if len(args) == 1:
            target = args[0]

        index    = new_variable('int',  target)
        iterator = new_variable('real', target)
        length   = new_variable('len',  target)
        # ...

        return Basic.__new__(cls, args, index, iterator, length)

    @property
    def arguments(self):
        return self._args[0]

    @property
    def index(self):
        return self._args[1]

    @property
    def iterator(self):
        return self._args[2]

    @property
    def length(self):
        return self._args[3]

    def __len__(self):
        return len(self.arguments)

class VariableGenerator(BasicGenerator):
    pass

class ZipGenerator(BasicGenerator):
    pass

class ProductGenerator(BasicGenerator):
    pass

def generator_as_block(generator, stmts, **kwargs):
    # ...
    settings = kwargs.copy()

    parallel = settings.pop('parallel', False)
    # ...

    # ...
    decs = []
    body = []
    # ...

    # TODO USE stmts

    # ...
    iterable = generator.arguments
    index    = generator.index
    iterator = generator.iterator
    length   = generator.length

    if not isinstance(iterable, (list, tuple, Tuple)):
        iterable = [iterable]

    if not isinstance(index, (list, tuple, Tuple)):
        index = [index]

    if not isinstance(iterator, (list, tuple, Tuple)):
        iterator = [iterator]

    if not isinstance(length, (list, tuple, Tuple)):
        length = [length]
    # ...

    # ...
    for n,xs in zip(length, iterable):
        decs += [Assign(n, Len(xs))]
    # ...

    # ...
    body += list(stmts)
    for i,n,x,xs in zip(index, length, iterator, iterable):

        body = [Assign(x, IndexedBase(xs.name)[i])] + body
        body = [For(i, Range(0, n), body, strict=False)]
    # ...

    if parallel:
        return ParallelBlock( decs, body )

    else:
        return SequentialBlock( decs, body )

#=========================================================================
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

#=========================================================================
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

        obj = Basic.__new__(cls, dtype, rank, is_stack_array, order, precision)
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

    def _sympystr(self, printer):
        sstr = printer.doprint
        return sstr(self.name)

    def view(self):
        """inspects the variable."""
        attributs = ','.join(i.view() for i in self.types)
        return 'TypeTuple({})'.format(attributs)

#=========================================================================
class TypeList(BasicTypeVariable):
    def __new__( cls, var ):
        assert(isinstance(var, (TypeVariable, TypeTuple, TypeList)))

        obj = Basic.__new__(cls, var)
        obj._tag = random_string( 4 )

        return obj

    @property
    def parent(self):
        return self._args[0]

    @property
    def name(self):
        return 'tl_{}'.format(self.tag)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return sstr(self.name)

    def view(self):
        """inspects the variable."""
        return 'TypeList({})'.format(self.parent.view())

#=========================================================================
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
        ls = [assign_type( i, rank=rank ) for i in expr.types]
        return assign_type( ls )

    else:
        raise TypeError('> wrong argument, given {}'.format(type(expr)))
