# coding: utf-8

import os
from os.path import join, dirname

from sympy import Symbol, Lambda, Function, Dummy
from sympy import Tuple, IndexedBase
from sympy.core.function import AppliedUndef
from sympy.core.function import UndefinedFunction
from sympy import Integer, Float
from sympy import sympify
from sympy import FunctionClass


from pyccel.codegen.utilities import random_string
from pyccel.ast.utilities import build_types_decorator
from pyccel.ast.core import Slice
from pyccel.ast.core import Variable, FunctionDef, Assign, AugAssign
from pyccel.ast.core import Return
from pyccel.ast.core  import For, Range, Len
from pyccel.ast.basic import Basic

from .datatypes import TypeVariable, TypeTuple, TypeList
from .semantic import Parser as SemanticParser
from .glossary import _internal_applications
from .glossary import _math_functions
from .glossary import _internal_map_functors

#=========================================================================
# some useful functions
# TODO add another argument to distinguish between len and other ints
def new_variable( dtype, var, tag = None, prefix=None, kind=None ):

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
class BasicGenerator(Basic):

    def __len__(self):
        return len(self.arguments)

#==============================================================================
class VariableGenerator(BasicGenerator):
    def __new__( cls, *args ):
        # ... create iterator and index variables
        target = args
        if len(args) == 1:
            target = args[0]

        tag = random_string( 4 )

        index    = new_variable('int',  target, tag = tag)
        iterator = new_variable('real', target, tag = tag)
        length   = new_variable('int',  target, tag = tag, kind='len')
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

#==============================================================================
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

        return Basic.__new__(cls, args, index, iterator, length, multi_index)

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

#==============================================================================
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

#    print('------------- BEFORE')
#    print(' iterable = ', iterable)
#    print(' index    = ', index   )
#    print(' iterator = ', iterator)
#    print(' length   = ', length  )

    # ...
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

            body = [For(i, Range(0, n), body, strict=False)]

    else:
        for i,n,x,xs in zip(index, length, iterator, iterable):

            if not isinstance(xs, (list, tuple, Tuple)):
                body = [Assign(x, IndexedBase(xs.name)[i])] + body

            else:
                for v in xs:
                    body = [Assign(x, IndexedBase(v.name)[i])] + body

            body = [For(i, Range(0, n), body, strict=False)]
    # ...

    if parallel:
        return ParallelBlock( decs, body )

    else:
        return SequentialBlock( decs, body )


#==============================================================================
# ...
def _attributs_from_type(t, d_var):
    if isinstance(t, TypeList):
        t = _attributs_from_type(t.parent, d_var)
        d_var['rank'] = d_var['rank'] + 1
        return t, d_var

    elif isinstance(t, TypeTuple):
        raise NotImplementedError()

    elif isinstance(t, TypeVariable):
        d_var['dtype']          = t.dtype
        d_var['rank']           = t.rank
        d_var['is_stack_array'] = t.is_stack_array
        d_var['order']          = t.order
        d_var['precision']      = t.precision

        return t, d_var
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
#==============================================================================

#==============================================================================
class AST(object):

    def __init__(self, parser, **kwargs):
        assert(isinstance(parser, SemanticParser))

        # ...
        self._expr            = parser.expr
        self._namespace       = parser.namespace
        self._d_types         = parser.d_types
        self._d_expr          = parser.d_expr
        self._tag             = parser.tag
        self.main             = parser.main
        self.main_type        = parser.main_type
        self._typed_functions = parser.typed_functions
        self._default_type    = parser.default_type
        # ...

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
    def namespace(self):
        return self._namespace

    @property
    def d_types(self):
        return self._d_types

    @property
    def d_expr(self):
        return self._d_expr

    @property
    def tag(self):
        return self._tag

    def inspect(self):
        print('============ types =============')
        print(self.d_types)
        for k,v in self.d_types.items():
            print('  {k} = {v}'.format(k=k, v=v.view()))
        print('================================')

    def _get_label(self, target, domain=False, codomain=False):
        # TODO improve
        if codomain:
            assert(not domain)
            if (isinstance(target, FunctionClass)):
                name = str(target)

            else:
                name = str(target.name)

            return name

        if domain:
            assert(not codomain)
            if (isinstance(target, FunctionClass)):
                name = str(target)

            else:
                name = str(target.name)

            _avail_funcs = list(self.typed_functions.keys()) + _math_functions
            if name in _avail_funcs:
                return name + '_args'

        if isinstance(target, FunctionDef):
            return str(target.name) + '_args'

        elif isinstance(target, UndefinedFunction):
            return str(target)

        elif isinstance(target, Symbol):
            return target.name

        else:
            raise NotImplementedError('for {}'.format(type(target)))

    def _get_type(self, target, domain=False, codomain=False):
        label = self._get_label(target, domain=domain, codomain=codomain)

        if label in self.d_types.keys():
            return self.d_types[label]

        return None

    def _set_type(self, target, value, domain=False, codomain=False):
        label = self._get_label(target, domain=domain, codomain=codomain)

        self.d_types[label] = value
        self._set_expr(value, target)

    def _set_expr(self, t_var, expr):
        self._d_expr[t_var.name] = expr

    def build_namespace(self):
        """builds the namespace from types."""
        raise NotImplementedError('')

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
            if name in _internal_map_functors:
                func, target = stmt.args

                # ... construct the generator
                generator = self._visit(target)
                if isinstance(generator, Variable):
                    generator = VariableGenerator(generator)
                # ...

                # ... construct the results
                type_codomain = self.main_type
                results = self._visit(type_codomain)

                # compute depth of the type list
                depth_out = len(list(type_codomain.atoms(TypeList)))
                print('>>> type  = ', type_codomain, type_codomain.view())
                print('>>> depth = ', depth_out)
                # ...

                # ...
                index    = generator.index
                iterator = generator.iterator
                # ...

                # ... list of all statements
                stmts = []
                # ...

                # ... use a multi index in the case of zip
                if isinstance(generator, ProductGenerator):

                    assert(isinstance(index, (list, tuple, Tuple)))

                    length = generator.length
                    if name == 'map':
                        multi_index = generator.multi_index

                        # TODO check formula
                        value = index[0]
                        for ix, nx in zip(index[1:], length[::-1][:-1]):
                            value = nx*value + ix

                        stmts += [Assign(multi_index, value)]

                        # update index to use multi index
                        index = multi_index
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
                    raise NotImplementedError()

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
                return generator_as_block( generator, stmts,
                                           parallel      = False )

            elif name == 'zip':
                arguments = [self._visit(i) for i in stmt.args]

                return ZipGenerator(*arguments)

            elif name == 'product':
                arguments = [self._visit(i) for i in stmt.args]

                return ProductGenerator(*arguments)

            else:
                raise NotImplementedError('')

        # Unknown object, we raise an error.
        raise TypeError('{node} not yet available'.format(node=type(stmt)))

    def _visit_Lambda(self, stmt):
        args = [self._visit(i) for i in stmt.variables]
        expr = self._visit(stmt.expr)
        # TODO improve
        results = self._visit(self.main)
        if not isinstance(results, (list, tuple, Tuple)):
            results = [results]

        # TODO improve
        body = [expr]

        if len(results) == 1:
            body += [Return(results[0])]

        else:
            body += [Return(results)]

        # ...
        decorators = {'types':         build_types_decorator(args),
                      'external_call': []}

        tag         = random_string( 6 )
        name      = 'lambda_{}'.format( tag )
        # ...

        return FunctionDef(name, args, results, body,
                           decorators=decorators)

        return expr

    def _visit_Symbol(self, stmt):
        t_var = self.d_types[stmt.name]
        d_var = _attributs_default()
        t_var, d_var = _attributs_from_type(t_var, d_var)

        dtype = d_var.pop('dtype')
        var = Variable( dtype, stmt.name, **d_var )

        return var

    def _visit_Integer(self, stmt):
        return stmt

    def _visit_Float(self, stmt):
        return stmt

    def _visit_TypeVariable(self, stmt):
        name  = 'dummy_{}'.format(stmt.tag)
        t_var = stmt

        d_var = _attributs_default()
        t_var, d_var = _attributs_from_type(t_var, d_var)

        dtype = d_var.pop('dtype')
        var = Variable( dtype, name, **d_var )

        return var

    def _visit_TypeTuple(self, stmt):
        # TODO
        name  = 'dummy_{}'.format(stmt.tag)
        t_var = stmt

        d_var = _attributs_default()
        t_var, d_var = _attributs_from_type(t_var, d_var)

        dtype = d_var.pop('dtype')
        var = Variable( dtype, name, **d_var )

        return var

    def _visit_TypeList(self, stmt):
        # TODO
        name  = 'dummy_{}'.format(stmt.tag)
        t_var = stmt

        d_var = _attributs_default()
        t_var, d_var = _attributs_from_type(t_var, d_var)

        dtype = d_var.pop('dtype')
        var = Variable( dtype, name, **d_var )

        return var

    def get_expr_from_type(self, t_var=None):
        if t_var is None:
            t_var = self.main_type

        return self.d_expr[t_var.name]
