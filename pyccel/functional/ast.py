# coding: utf-8

import os
from os.path import join, dirname

from sympy import Symbol, Lambda, Function, Dummy
from sympy import Tuple, IndexedBase, Indexed
from sympy.core.function import AppliedUndef
from sympy.core.function import UndefinedFunction
from sympy import Integer, Float
from sympy import sympify
from sympy import FunctionClass


from pyccel.codegen.utilities import random_string
from pyccel.ast.utilities import build_types_decorator
from pyccel.ast.core import Slice
from pyccel.ast.core import Variable, FunctionDef, Assign, AugAssign
from pyccel.ast.core import Return, Pass, Import, String, FunctionCall
from pyccel.ast.core  import For, Range, Len, Print
from pyccel.ast.datatypes import get_default_value
from pyccel.ast.datatypes import NativeInteger, NativeReal, NativeComplex, NativeBool
from pyccel.ast.basic import Basic

from pyccel.ast.parallel.openmp import OMP_For, OMP_Private, OMP_Parallel
from pyccel.ast.parallel.openmp import OMP_Schedule
from pyccel.ast.parallel.openmp import OMP_NumThread
from pyccel.ast.parallel.openmp import OMP_Reduction

from .datatypes import TypeVariable, TypeTuple, TypeList
from .semantic import Parser as SemanticParser
from .glossary import _internal_applications
from .glossary import _math_functions
from .glossary import _internal_map_functors

#========================================================================
# TODO improve or copy from pyccel.parser
def _get_name(i):
    if isinstance(i, Symbol):
        return i.name

    elif isinstance(i, IndexedBase):
        return str(i)

    elif isinstance(i, Indexed):
        return _get_name(i.base)

    else:
        raise NotImplementedError()

#========================================================================
# some useful functions
# TODO add another argument to distinguish between len and other ints
def new_variable( dtype, var, tag = None, prefix = None, kind = None ):

    # ...
    if prefix is None:
        prefix = ''

    _prefix = '{}'.format(prefix)
    # ...

    # ...
    if dtype == 'int':
        if kind == 'len':
            _prefix = 'n{}'.format(_prefix)

        elif kind == 'multi':
            _prefix = 'im{}'.format(_prefix)

        else:
            _prefix = 'i{}'.format(_prefix)

    elif dtype == 'real':
        _prefix = 'r{}'.format(_prefix)

    else:
        raise NotImplementedError()
    # ...

    # ...
    if tag is None:
        tag = random_string( 4 )
    # ...

    pattern = '{prefix}{dim}_{tag}'
    _print = lambda d,t: pattern.format(prefix=_prefix, dim=d, tag=t)

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

    elif isinstance(var, (list, tuple, Tuple)):
        ls = [new_variable( dtype, x,
                            tag = tag,
                            prefix = str(i),
                            kind = kind )
              for i,x in enumerate(var)]
        return Tuple(*ls)

    else:
        raise NotImplementedError('{} not available'.format(type(var)))


#==============================================================================
def _get_default_value(var, op=None):
    """Returns the default value of a variable depending on its datatype and the
    used operation."""
    dtype = var.dtype
    if op is None:
        return get_default_value(dtype)

    if isinstance(dtype, NativeInteger):
        if op == '*':
            return 1

        else:
            return 0

    elif isinstance(dtype, NativeReal):
        if op == '*':
            return 1.0

        else:
            return 0.0

    elif isinstance(dtype, NativeComplex):
        # TODO check if this fine with pyccel
        if op == '*':
            return 1.0

        else:
            return 0.0

#    elif isinstance(dtype, NativeBool):

    raise NotImplementedError('TODO')

#=========================================================================
class LambdaFunctionDef(FunctionDef):

    """."""
    def __new__( cls, name, arguments, results, body, **kwargs ):
        generators = kwargs.pop('generators', {})
        m_results  = kwargs.pop('m_results',   [])

        obj = FunctionDef.__new__(cls, name, arguments, results, body, **kwargs)
        obj._generators = generators
        obj._m_results  = m_results

        return obj

    @property
    def generators(self):
        return self._generators

    @property
    def m_results(self):
        return self._m_results


#=========================================================================
class Shaping(Basic):

    def __new__( cls, generator ):

        # ... define the name for the shape and the statements to be able to
        #     compute it inside the python interface

        if isinstance(generator, VariableGenerator):
            var   = generator.length
            stmts = [Assign(var, Len(generator.arguments))]

        elif isinstance(generator, ZipGenerator):
            var   = generator.length
            stmts = [Assign(var, Len(generator.arguments[0]))]

        elif isinstance(generator, ProductGenerator):
            arguments = generator.arguments
            length    = generator.length

            stmts = [Assign(l, Len(a)) for l,a in zip(length, arguments)]

            if generator.is_list:
                n = 1
                for i in length:
                    n *= i

            else:
                n = length

            var = Dummy()
            stmts += [Assign(var, n)]

        else:
            msg = 'not available for {}'.format(type(generator))
            raise NotImplementedError(msg)

        stmts = Tuple(*stmts)

        return Basic.__new__(cls, var, stmts)

    @property
    def var(self):
        return self._args[0]

    @property
    def stmts(self):
        return self._args[1]

#=========================================================================
class BasicGenerator(Basic):

    def __len__(self):
        return len(self.arguments)

#==============================================================================
class VariableGenerator(BasicGenerator):
    def __new__( cls, *args ):
        # ... create iterator and index variables
        iterable = args
        if len(args) == 1:
            iterable = args[0]

        tag = random_string( 4 )

        index    = new_variable('int',  iterable, tag = tag)
        iterator = new_variable('real', iterable, tag = tag)
        length   = new_variable('int',  iterable, tag = tag, kind='len')
        # ...

        return Basic.__new__( cls, iterable, index, iterator, length )

    @property
    def arguments(self):
        # TODO change name to iterable
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

    @property
    def private(self):
        # TODO add iterable?
        args = [self.index, self.iterator]
        return Tuple(*args)

#=========================================================================
class ZipGenerator(BasicGenerator):
    def __new__( cls, *args ):
        # ... create iterator and index variables
        target = args
        if len(args) == 1:
            print(args)
            raise NotImplementedError('')

        tag = random_string( 4 )

        length      = new_variable('int',  target[0], tag = tag, kind='len')
        index       = new_variable('int',  target[0], tag = tag)
        iterator    = new_variable('real', target, tag = tag)
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

    @property
    def private(self):
        # TODO add iterable?
        args = [self.index] + list(self.iterator)
        return Tuple(*args)

#==============================================================================
class ProductGenerator(BasicGenerator):

    def __new__( cls, *args ):
        # ... create iterator and index variables
        target = args
        assert(len(args) > 1)

        tag = random_string( 4 )

        length      = new_variable('int',  target, tag = tag, kind='len')
        index       = new_variable('int',  target, tag = tag)
        iterator    = new_variable('real', target, tag = tag)
        multi_index = new_variable('int',  target[0], tag = tag, kind='multi')
        # ...

        obj = Basic.__new__(cls, args, index, iterator, length, multi_index)
        obj._is_list = False

        return obj

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

    @property
    def multi_index(self):
        return self._args[4]

    @property
    def private(self):
        # TODO add iterable?
        args = list(self.index) + list(self.iterator)
        if self.is_list:
            args += [self.multi_index]
        return Tuple(*args)

    @property
    def is_list(self):
        return self._is_list

    def set_as_list(self):
        self._is_list = True

#=========================================================================
class Reduction(Basic):
    def __new__( cls, op, arguments ):
#        assert( isinstance( generator, BasicGenerator ) )
        assert( isinstance( op, str ) )
        assert( op in ['+', '-'] )

        return Basic.__new__( cls, op, arguments )

    @property
    def op(self):
        return self._args[0]

    @property
    def arguments(self):
        return self._args[1]


#=========================================================================
class BasicBlock(Basic):
    pass

#=========================================================================
class MainBlock(BasicBlock):
    """."""
    def __new__( cls, block, **kwargs ):

        # ...
        settings = kwargs.copy()

        accelerator = settings.pop('accelerator', None)
        # ...

        # ...
        assert( isinstance( block, GeneratorBlock ) )
        # ...

        # ...
        decs = block.decs
        body = block.body
        # ...

        # ... add parallel loop
        if accelerator:
            if not( accelerator == 'omp' ):
                raise NotImplementedError('Only OpenMP is available')

            # ... create clauses
            clauses = []

#            ##### DEBUG
#            clauses += [OMP_NumThread(4)]
            # ...

            # ... create variables
            variables = []
            # ...

            # ...
            body = [OMP_Parallel( clauses, variables, body )]
            # ...

            # TODO this is a hack to handle the last comment after a loop, so that
            #      it can be parsed and added to the For
            body += [Pass()]
        # ...

        decs = Tuple(*decs)
        body = Tuple(*body)

        return Basic.__new__( cls, decs, body )

    @property
    def decs(self):
        return self._args[0]

    @property
    def body(self):
        return self._args[1]


#=========================================================================
class GeneratorBlock(BasicBlock):
    """."""
    def __new__( cls, generator, stmts, **kwargs ):

        # ...
        settings = kwargs.copy()

        accelerator = settings.pop('accelerator')
        nowait      = settings.pop('nowait')
        schedule    = settings.pop('schedule')
        chunk       = settings.pop('chunk')
        # ...

        # ...
        assert(isinstance(generator, BasicGenerator))

        decs, body = _build_block( generator, stmts )
        # ...

        # ...
        assert(isinstance(decs, (tuple, list, Tuple)))
        assert(isinstance(body, (tuple, list, Tuple)))

        decs = Tuple(*decs)
        body = Tuple(*body)
        # ...

        # ... define private variables of the current block
        private_vars = generator.private
        # TODO add private vars from internal blocks
        private_vars = Tuple(*private_vars)
        # ...

        # ... add parallel loop
        if accelerator:
            if not( accelerator == 'omp' ):
                raise NotImplementedError('Only OpenMP is available')

            # ... create clauses
            clauses = []

            # TODO move this treatment to OMP_Schedule
            if chunk is None:
                clauses += [OMP_Schedule(schedule)]

            else:
                clauses += [OMP_Schedule(schedule, chunk)]

            if private_vars:
                clauses += [OMP_Private(*private_vars)]
            # ...

            # ...
            assert(len(body) == 1)
            loop = body[0]
            body = [OMP_For(loop, Tuple(*clauses), nowait)]
            # ...
        # ...

        decs = Tuple(*decs)
        body = Tuple(*body)

        return Basic.__new__( cls, generator, decs, body, private_vars )

    @property
    def generator(self):
        return self._args[0]

    @property
    def decs(self):
        return self._args[1]

    @property
    def body(self):
        return self._args[2]

    @property
    def private(self):
        return self._args[3]


#=========================================================================
class ReductionGeneratorBlock(GeneratorBlock):
    """."""
    def __new__( cls, block, reduction, lhs, **kwargs ):

        # ...
        settings = kwargs.copy()

        accelerator = settings.pop('accelerator')
        # ...

        assert( isinstance( reduction, Reduction ) )

        # ...
        if isinstance(lhs, (list, tuple, Tuple)):
            raise NotImplementedError()
        # ...

        # ...
        generator    = block.generator
        decs         = block.decs
        body         = block.body
        private_vars = generator.private
        # ...

        # ...
        lhs_name = lhs.name

        assign_stmts = list(block.body.atoms(Assign))
        # ...

        # ...
        ls = [i for i in assign_stmts if _get_name(i.lhs) == lhs_name]
        if len(ls) > 1:
            raise NotImplementedError()

        assign_iterable = ls[0]
        rhs = assign_iterable.rhs
        new_stmt = AugAssign(lhs, reduction.op, rhs)

        body = block.body.subs(assign_iterable, new_stmt)

        decs = block.decs
        # ...

        # ... add initial values for reduced variables
        _reduction_init = lambda i: _get_default_value( i, op=reduction.op )
        reduction_stmts = [Assign(r, _reduction_init(r)) for r in reduction.arguments]

        # update declarations
        decs = list(decs) + reduction_stmts
        decs = Tuple(*decs)
        # ...

        # ... define private variables of the current block
        # TODO add private vars from internal blocks
        private_vars = Tuple(*private_vars)
        # ...

        # ... add parallel loop
        if accelerator:
            if not( accelerator == 'omp' ):
                raise NotImplementedError('Only OpenMP is available')

            # ...
            omp_fors = list(body.atoms(OMP_For))
            omp_for = [i for i in body if isinstance(i, OMP_For)]
            assert(len(omp_for) == 1)
            omp_for = omp_for[0]

            loop    = omp_for.loop
            clauses = list(omp_for.clauses)
            nowait  = omp_for.nowait

            # ...
            reduction_args = [reduction.op] + list(reduction.arguments)
            clauses += [OMP_Reduction( *reduction_args ) ]
            clauses = Tuple(*clauses)
            # ...

            new = OMP_For(loop, clauses, nowait)
            body = body.subs(omp_for, new)
        # ...

        decs = Tuple(*decs)
        body = Tuple(*body)

        return Basic.__new__( cls, generator, decs, body, private_vars )

    @property
    def generator(self):
        return self._args[0]

    @property
    def decs(self):
        return self._args[1]

    @property
    def body(self):
        return self._args[2]

    @property
    def private(self):
        return self._args[3]

#==========================================================================
def _build_block( generator, stmts ):

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

#    print('------------- BEFORE')
#    print(' iterable = ', iterable)
#    print(' index    = ', index   )
#    print(' iterator = ', iterator)
#    print(' length   = ', length  )

    # ... TODO  use shape_stmts
    for n,xs in zip(length, iterable):
        decs += [Assign(n, Len(xs))]
    # ...

    # ... append the same length for product
    if isinstance(generator, ProductGenerator):
        length = length*len(generator)
    # ...

#    print('------------- AFTER')
#    print(' iterable = ', iterable)
#    print(' index    = ', index   )
#    print(' iterator = ', iterator)
#    print(' length   = ', length  )

    # ...
    body += list(stmts)
    # ...

    # ...
    if isinstance(generator, ZipGenerator):
        for i,n in zip(index, length):
            for x,xs in zip(iterator, iterable):

                if not isinstance(xs, (list, tuple, Tuple)):
                    body = [Assign(x, IndexedBase(xs.name)[i])] + body

                else:
                    for v in xs:
                        body = [Assign(x, IndexedBase(v.name)[i])] + body

            body = [For(i, Range(0, n), Tuple(*body), strict=False)]

    else:
        for i,n,x,xs in zip(index, length, iterator, iterable):

            if not isinstance(xs, (list, tuple, Tuple)):
                body = [Assign(x, IndexedBase(xs.name)[i])] + body

            else:
                for v in xs:
                    body = [Assign(x, IndexedBase(v.name)[i])] + body

            body = [For(i, Range(0, n), Tuple(*body), strict=False)]
    # ...

    return decs, body


#==========================================================================
# ...
def _attributs_from_type(t, d_var):

    if not isinstance(t, (TypeVariable, Variable)):
        msg = '> Expecting TypeVariable or Variable, but {} was given'.format(type(t))
        raise TypeError(msg)

    d_var['dtype']          = t.dtype
    d_var['rank']           = t.rank
    d_var['is_stack_array'] = t.is_stack_array
    d_var['order']          = t.order
    d_var['precision']      = t.precision
    d_var['shape']          = t.shape

    return d_var
# ...

# ... default values
def _attributs_default():
    d_var = {}

    d_var['dtype']          = None
    d_var['rank']           = 0
    d_var['allocatable']    = False
    d_var['is_stack_array'] = False
    d_var['is_pointer']     = False
    d_var['is_target']      = False
    d_var['shape']          = None
    d_var['order']          = 'C'
    d_var['precision']      = None

    return d_var
# ...
#==========================================================================

#==========================================================================
class AST(object):

    def __init__(self, parser, **kwargs):
        assert(isinstance(parser, SemanticParser))

        # ...
        self._expr            = parser.expr
        self._d_types         = parser.d_types
        self._d_domain_types  = parser.d_domain_types
        self._d_expr          = parser.d_expr
        self._tag             = parser.tag
        self.main             = parser.main
        self.main_type        = parser.main_type
        self._typed_functions = parser.typed_functions
        self._default_type    = parser.default_type
        self._generators      = {}
        # ...

        # ...
        settings = kwargs.copy()

        accelerator = settings.pop('accelerator', None)
        # ...

        # ...
        if accelerator is None:
            accelerator = 'omp'

        assert(isinstance(accelerator, str))
        assert(accelerator in ['openmp', 'openacc', 'omp', 'acc'])

        if accelerator == 'openmp':
            accelerator = 'omp'

        elif accelerator == 'openacc':
            accelerator = 'acc'

        self._accelerator = accelerator
        # ...

        # ...
        self._nowait   = settings.pop('nowait', True)
        self._schedule = settings.pop('schedule', 'static')
        self._chunk    = settings.pop('chunk', None)
        # ...

#        print('------------------')
#        print('> d_types ')
#        self.inspect()
#        print('INITIAL TYPE = ', self.main_type)
#        print('')

    @property
    def expr(self):
        return self._expr

    @property
    def typed_functions(self):
        return self._typed_functions

    @property
    def default_type(self):
        return self._default_type

    @property
    def d_types(self):
        return self._d_types

    @property
    def d_domain_types(self):
        return self._d_domain_types

    @property
    def d_expr(self):
        return self._d_expr

    @property
    def tag(self):
        return self._tag

    @property
    def generators(self):
        return self._generators

    @property
    def accelerator(self):
        return self._accelerator

    @property
    def schedule(self):
        return self._schedule

    @property
    def chunk(self):
        return self._chunk

    @property
    def nowait(self):
        return self._nowait

    def inspect(self):
        print(self.d_types)
        for k,v in self.d_types.items():
            print('  {k} = {v}'.format(k=k, v=v.view()))

    def set_generator(self, results, generator):
        self._generators[results] = generator

    def doit(self):
        return self._visit(self.expr)

    def _visit(self, stmt=None):

        if stmt is None:
            stmt = self.expr

        cls = type(stmt)
        name = cls.__name__

        method = '_visit_{}'.format(name)
        if hasattr(self, method):
            return getattr(self, method)(stmt)

        elif name in _internal_applications:

#            print('----- types ')
#            print(self.d_types)
#            print('----- associations ')
#            print(self.d_domain_types)

#            print('>>>>>>> {}'.format(name))
#            print('> codomain :: ', type_codomain)
#            print('              ', type_codomain.view())
#            print('> domain   :: ', type_domain)
#            print('              ', type_domain.view())
#            print('')

            if name == 'map':
                return self._visit_functor_map(stmt)

            elif name == 'xmap':
                return self._visit_functor_xmap(stmt)

            elif name == 'tmap':
                return self._visit_functor_tmap(stmt)

            elif name == 'reduce':
                return self._visit_functor_reduce(stmt)

            elif name == 'zip':
                self.main_type = type_domain
                arguments = [self._visit(i) for i in stmt.args]

                return ZipGenerator(*arguments)

            elif name == 'product':
                # TODO fix bug
#                self.main_type = type_domain
                arguments = [self._visit(i) for i in stmt.args]

                return ProductGenerator(*arguments)

            else:
                raise NotImplementedError('')

        # Unknown object, we raise an error.
        raise TypeError('{node} not yet available'.format(node=type(stmt)))

    def _visit_functor_map(self, stmt):
        func   = stmt.args[0]
        args   = stmt.args[1:]

        # ... get the codomain type
        type_codomain  = self.main_type
        type_domain    = self.d_domain_types[type_codomain]
        # ...

        # ... construct the generator
        target = [self._visit(i) for i in args]

        if len(target) == 1:
            target = target[0]
            assert( isinstance(target, Variable) )

            generator = VariableGenerator(target)

        else:
            generator = ZipGenerator(*target)
        # ...

        # ... construct the results
        results = self._visit(type_codomain)

        # compute depth of the type list
        # TODO do we still need this?
        depth_out = len(list(type_codomain.atoms(TypeList)))
        # ...

        # ...
        index    = generator.index
        iterator = generator.iterator
        # ...

        # ... list of all statements
        stmts = []
        # ...

        # ... we set the generator after we treat map/tmap
        self.set_generator(results, generator)
        # ...

        # ... apply the function to arguments
        if isinstance(iterator, Tuple):
            rhs = func( *iterator )

        else:
            rhs = func( iterator )
        # ...

        # ... create lhs
        lhs = generator.iterator
        # TODO check this
        if isinstance(lhs, Tuple) and len(lhs) == 1:
            lhs = lhs[0]
        # ...

        # ... create lhs for storing the result
        if isinstance(results, Variable):
            results = [results]

        else:
            msg = '{} not available'.format(type(results))
            raise NotImplementedError(msg)

        if not isinstance(index, Tuple):
            index = [index]

        else:
            index = list([i for i in index])

        lhs = []
        for r in results:
            m = r.rank - depth_out
            ind = index + [Slice(None, None)] * m
            if len(ind) == 1:
                ind = ind[0]

            lhs.append(IndexedBase(r.name)[ind])

        lhs = Tuple(*lhs)
        if len(lhs) == 1:
            lhs = lhs[0]
        # ...

        # ... create core statement
        stmts += [Assign(lhs, rhs)]
        # ...

        # TODO USE THIS
#                expr = self.get_expr_from_type()

        # return the associated for loops
        return GeneratorBlock ( generator, stmts,
                                accelerator = self.accelerator,
                                nowait      = self.nowait,
                                schedule    = self.schedule,
                                chunk       = self.chunk )

    def _visit_functor_xmap(self, stmt):
        func   = stmt.args[0]
        args   = stmt.args[1:]

        assert(len(args) > 1)

        # ... get the codomain type
        type_codomain  = self.main_type
        type_domain    = self.d_domain_types[type_codomain]
        # ...

        # ... construct the generator
        target = [self._visit(i) for i in args]

        generator = ProductGenerator(*target)
        # ...

        # ... construct the results
        results = self._visit(type_codomain)

        # compute depth of the type list
        # TODO do we still need this?
        depth_out = len(list(type_codomain.atoms(TypeList)))
        # ...

        # ...
        index    = generator.index
        iterator = generator.iterator
        # ...

        # ... list of all statements
        stmts = []
        # ...

        # ... use a multi index in the case of zip
        length      = generator.length
        multi_index = generator.multi_index
        generator.set_as_list()

        # TODO check formula
        value = index[0]
        for ix, nx in zip(index[1:], length[::-1][:-1]):
            value = nx*value + ix

        stmts += [Assign(multi_index, value)]

        # update index to use multi index
        index = multi_index
        # ...

        # ... we set the generator after we treat map/tmap
        self.set_generator(results, generator)
        # ...

        # ... apply the function to arguments
        if isinstance(iterator, Tuple):
            rhs = func( *iterator )

        else:
            rhs = func( iterator )
        # ...

        # ... create lhs
        lhs = generator.iterator
        # TODO check this
        if isinstance(lhs, Tuple) and len(lhs) == 1:
            lhs = lhs[0]
        # ...

        # ... create lhs for storing the result
        if isinstance(results, Variable):
            results = [results]

        else:
            msg = '{} not available'.format(type(results))
            raise NotImplementedError(msg)

        if not isinstance(index, Tuple):
            index = [index]

        else:
            index = list([i for i in index])

        lhs = []
        for r in results:
            m = r.rank - depth_out
            ind = index + [Slice(None, None)] * m
            if len(ind) == 1:
                ind = ind[0]

            lhs.append(IndexedBase(r.name)[ind])

        lhs = Tuple(*lhs)
        if len(lhs) == 1:
            lhs = lhs[0]
        # ...

        # ... create core statement
        stmts += [Assign(lhs, rhs)]
        # ...

        # TODO USE THIS
#                expr = self.get_expr_from_type()

        # return the associated for loops
        return GeneratorBlock ( generator, stmts,
                                accelerator = self.accelerator,
                                nowait      = self.nowait,
                                schedule    = self.schedule,
                                chunk       = self.chunk )

    def _visit_functor_tmap(self, stmt):
        func   = stmt.args[0]
        args   = stmt.args[1:]

        assert(len(args) > 1)

        # ... get the codomain type
        type_codomain  = self.main_type
        type_domain    = self.d_domain_types[type_codomain]
        # ...

        # ... construct the generator
        target = [self._visit(i) for i in args]

        generator = ProductGenerator(*target)
        # ...

        # ... construct the results
        results = self._visit(type_codomain)

        # compute depth of the type list
        # TODO do we still need this?
        depth_out = len(list(type_codomain.atoms(TypeList)))
        # ...

        # ...
        index    = generator.index
        iterator = generator.iterator
        # ...

        # ... list of all statements
        stmts = []
        # ...

        # ... we set the generator after we treat map/tmap
        self.set_generator(results, generator)
        # ...

        # ... apply the function to arguments
        if isinstance(iterator, Tuple):
            rhs = func( *iterator )

        else:
            rhs = func( iterator )
        # ...

        # ... create lhs
        lhs = generator.iterator
        # TODO check this
        if isinstance(lhs, Tuple) and len(lhs) == 1:
            lhs = lhs[0]
        # ...

        # ... create lhs for storing the result
        if isinstance(results, Variable):
            results = [results]

        else:
            msg = '{} not available'.format(type(results))
            raise NotImplementedError(msg)

        if not isinstance(index, Tuple):
            index = [index]

        else:
            index = list([i for i in index])

        lhs = []
        for r in results:
            m = r.rank - depth_out
            ind = index + [Slice(None, None)] * m
            if len(ind) == 1:
                ind = ind[0]

            lhs.append(IndexedBase(r.name)[ind])

        lhs = Tuple(*lhs)
        if len(lhs) == 1:
            lhs = lhs[0]
        # ...

        # ... create core statement
        stmts += [Assign(lhs, rhs)]
        # ...

        # TODO USE THIS
#                expr = self.get_expr_from_type()

        # return the associated for loops
        return GeneratorBlock ( generator, stmts,
                                accelerator = self.accelerator,
                                nowait      = self.nowait,
                                schedule    = self.schedule,
                                chunk       = self.chunk )

    def _visit_functor_reduce(self, stmt):
        arguments = stmt.args

        assert( len(arguments) == 2 )
        op     = arguments[0]
        target = arguments[1]

        # TODO
        op = '+'

        # ... get the codomain type
        type_codomain  = self.main_type
        type_domain    = self.d_domain_types[type_codomain]
        # ...

        # ... construct the results
        results = self._visit(type_codomain)

        # compute depth of the type list
        # TODO do we still need this?
        depth_out = 0
        # ...

        # ... construct the generator
        block = self._visit(target)
        assert( isinstance( block, GeneratorBlock ) )

        generator = block.generator
        if isinstance(generator, Variable):
            generator = VariableGenerator(generator)

        assert( isinstance( generator, BasicGenerator ) )
        # ...

        # ...
        index    = generator.index
        iterator = generator.iterator
        # ...

        # ... list of all statements
        stmts = []
        # ...

        # ... we set the generator after we treat map/tmap
        self.set_generator(results, generator)
        # ...

        # ... create lhs
        lhs = generator.iterator
        # TODO check this
        if isinstance(lhs, Tuple) and len(lhs) == 1:
            lhs = lhs[0]
        # ...

        # ... create lhs for storing the result
        if isinstance(results, Variable):
            results = [results]

        else:
            msg = '{} not available'.format(type(results))
            raise NotImplementedError(msg)

        if not isinstance(index, Tuple):
            index = [index]

        else:
            index = list([i for i in index])

        lhs = []
        for r in results:
            m = r.rank - depth_out
            ind = [Slice(None, None)] * m
            if ind:
                if len(ind) == 1:
                    ind = ind[0]

                lhs.append(IndexedBase(r.name)[ind])

            else:
                lhs.append(Symbol(r.name))

        lhs = Tuple(*lhs)
        if len(lhs) == 1:
            lhs = lhs[0]
        # ...

        # ... add reduction
        reduction = Reduction( op, results )
        # ...

        # TODO USE THIS
#                expr = self.get_expr_from_type()

        # return the associated for loops
        return ReductionGeneratorBlock ( block, reduction, lhs,
                                         accelerator = self.accelerator )


    def _visit_Lambda(self, stmt):
        # ...
        args = [self._visit(i) for i in stmt.variables]
        # ...

        # ...
        body = self._visit(stmt.expr)
        body = MainBlock( body,
                          accelerator = self.accelerator )
        body = [body]
        # ...

#        # ... DEBUG
#        body += [Import('omp_get_max_threads', 'pyccel.stdlib.internal.openmp')]
#
#        msg = lambda x: (String('> maximum available threads = '), x)
#        x = FunctionCall('omp_get_max_threads', ())
#        body += [Print(msg(x))]
#        # ...

        # ...
        results = self._visit(self.main)
        if not isinstance(results, (list, tuple, Tuple)):
            results = [results]
        # ...

        # ... scalar results
        s_results = [r for r in results if r.rank == 0]
        # ...

        # ... vector/matrix results as inout arguments
        m_results = [r for r in results if not r in s_results]
        # ...

        # ... return a function def where
        #     we append m_results to the arguments as inout
        #     and we return all results.
        #     first, we initialize arguments_inout to False for all args
        inout  = [False for i in args]
        inout += [True for i in m_results]

        args = args + m_results
        # ...

        # ...
        if len(s_results) == 1:
            body += [Return(s_results[0])]

        elif len(results) > 1:
            body += [Return(s_results)]
        # ...

        # ...
#        decorators = {'types':         build_types_decorator(args),
#                      'external_call': []}

        decorators = {'types':         build_types_decorator(args),
                      'external': []}

        tag         = random_string( 6 )
        name      = 'lambda_{}'.format( tag )
        # ...

        return LambdaFunctionDef( name, args, s_results, body,
                                  arguments_inout = inout,
                                  decorators      = decorators,
                                  generators      = self.generators,
                                  m_results       = m_results )

    def _visit_Integer(self, stmt):
        return stmt

    def _visit_Float(self, stmt):
        return stmt

    def _visit_Symbol(self, stmt):
        t_var = self.d_types[stmt.name]
        return self._visit(t_var)

    def _visit_TypeVariable(self, stmt):
        t_var = stmt

        d_var = _attributs_default()
        d_var = _attributs_from_type(t_var, d_var)

        dtype = d_var.pop('dtype')
        name  = 'dummy_{}'.format(stmt.tag)
        var   = Variable( dtype, name, **d_var )

        return var

    def _visit_TypeTuple(self, stmt):
        ls = []
        for e,t_var in enumerate(stmt.types):
            d_var = _attributs_default()
            d_var = _attributs_from_type(t_var, d_var)

            dtype = d_var.pop('dtype')
            name  = 'dummy_{}_{}'.format(e, stmt.tag)
            var   = Variable( dtype, name, **d_var )

            ls.append(var)

        var = Tuple(*ls)
        if len(var) == 1:
            return var[0]

        else:
            return var

    def _visit_TypeList(self, stmt):
        t_var = stmt

        rank = len(stmt)
        var = self._visit(stmt.types)

        if isinstance(var, Tuple):
            ls = []
            for e,v in enumerate(var):
                d_var = _attributs_default()
                d_var = _attributs_from_type(v, d_var)
                d_var['rank'] += rank

                dtype = d_var.pop('dtype')
                name  = 'dummy_{}_{}'.format(e, stmt.tag)
                var   = Variable( dtype, name, **d_var )

                ls.append(var)

            return Tuple(*ls)

        elif isinstance(var, Variable):
            d_var = _attributs_default()
            d_var = _attributs_from_type(var, d_var)
            d_var['rank'] += rank

            dtype = d_var.pop('dtype')
            name  = 'dummy_{}'.format(stmt.tag)
            var = Variable( dtype, name, **d_var )

            return var

        else:
            msg = 'Expecting a Tuple or Variable, but {} was given'
            msg = msg.format(type(var))
            raise TypeError(msg)

    def _visit_GeneratorBlock(self, stmt):
        return stmt

    def get_expr_from_type(self, t_var=None):
        if t_var is None:
            t_var = self.main_type

        return self.d_expr[t_var.name]
