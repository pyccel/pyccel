# -*- coding: UTF-8 -*-

# TODO use OrderedDict when possible

import os
import sys
import importlib
import numpy as np
from types import FunctionType

from sympy import Indexed, IndexedBase, Tuple, Lambda
from sympy.core.function import AppliedUndef
from sympy.core.function import UndefinedFunction

from pyccel.codegen.utilities import construct_flags as construct_flags_pyccel
from pyccel.codegen.utilities import execute_pyccel
from pyccel.codegen.utilities import get_source_function
from pyccel.codegen.utilities import random_string
from pyccel.codegen.utilities import write_code
from pyccel.codegen.utilities import mkdir_p
from pyccel.ast.datatypes import dtype_and_precsision_registry as dtype_registry
from pyccel.ast import Variable, Len, Assign, AugAssign
from pyccel.ast import For, Range, FunctionDef
from pyccel.ast import FunctionCall
from pyccel.ast import Comment, AnnotatedComment
from pyccel.ast import Print, Pass, Return
from pyccel.ast import ListComprehension
from pyccel.ast.datatypes import NativeInteger, NativeReal, NativeComplex, NativeBool
from pyccel.codegen.printing.pycode import pycode
from pyccel.codegen.printing.fcode  import fcode
from pyccel.ast.utilities import build_types_decorator
from pyccel.ast.datatypes import get_default_value
from pyccel.parser import Parser

#==============================================================================
_accelerator_registery = {'openmp':  'omp',
                          'openacc': 'acc',
                          None:      None}

_known_unary_functions = {'sum': '+',
                          'add': '+',
                          'mul': '*',
                                   }

_known_binary_functions = {}

_known_functions  = dict(_known_unary_functions, **_known_binary_functions)

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


#==============================================================================
def _extract_core_expr(expr):
    """extract core expression from a lambda expression"""
    if isinstance(expr, Lambda):
        return _extract_core_expr(expr.expr)

    elif isinstance(expr, ListComprehension):
        return _extract_core_expr(expr.expr)

    elif isinstance(expr, AppliedUndef):
        name = expr.__class__.__name__
        if name in _known_functions.keys():
            args = expr.args
            args = [_extract_core_expr(i) for i in args]
            if len(args) == 1:
                return args[0]

            else:
                raise NotImplementedError('')

        else:
            return expr

    else:
        raise NotImplementedError('{} not implemented'.format(type(expr)))

#==============================================================================
class VisitorLambda(object):
    """A visitor class to allow for manipulatin a lambda expression."""

    def __init__(self, expr, **kwargs):
        assert(isinstance(expr, Lambda))

        self._expr = expr
        self._core = _extract_core_expr(expr.expr)
        self.rank = 0

        self._dependencies   = kwargs.pop('dependencies', {})
        self._dependencies_code = kwargs.pop('dependencies_code', None)
        # TODO allow to reconstruct the dep code from namespace

        self._accelerator = kwargs.pop('accelerator', None)
        self._parallel    = kwargs.pop('parallel', None)
        self._inline      = kwargs.pop('inline', False)
        self._schedule    = kwargs.pop('schedule', None)

        if self.accelerator == 'openmp':
            if self.schedule is None:
                self._schedule = 'static'

            if self.parallel is None:
                self._parallel = True

        self._iterators = []
        self._iterables = []

        self._op = None

    @property
    def expr(self):
        return self._expr

    @property
    def variables(self):
        return self.expr.variables

    @property
    def core(self):
        return self._core

    @property
    def dependencies(self):
        return self._dependencies

    @property
    def dependencies_code(self):
        return self._dependencies_code

    @property
    def accelerator(self):
        return self._accelerator

    @property
    def parallel(self):
        return self._parallel

    @property
    def inline(self):
        return self._inline

    @property
    def schedule(self):
        return self._schedule

    @property
    def iterators(self):
        return self._iterators

    @property
    def iterables(self):
        return self._iterables

    @property
    def op(self):
        return self._op

    def insert_iterator(self, x):
        if isinstance(x, (tuple, list, Tuple)):
            self._iterators += list([i for i in x])

        else:
            self._iterators.append(x)

    def insert_iterable(self, x):
        if isinstance(x, (tuple, list, Tuple)):
            self._iterables += list([i for i in x])

        else:
            self._iterables.append(x)

    def _set_op(self, op):
        self._op = _known_functions[op]

    def _visit(self, stmt):

        cls = type(stmt)
        name = cls.__name__

        method = '_visit_{}'.format(name)
        if hasattr(self, method):
            return getattr(self, method)(stmt)

        elif name in _known_functions.keys():
            self._set_op(name)
            # TODO must reset op
            args = stmt.args
            if name in _known_unary_functions.keys():
                assert(len(args) == 1)
                args = args[0]

            else:
                raise NotImplementedError('')

#            print('arg = ', args)
#            import sys; sys.exit(0)
            return self._visit(args)

        # Unknown object, we raise an error.
        raise TypeError('{node} not yet available'.format(node=type(stmt)))

    def _visit_Lambda(self, stmt):
        return self._visit(stmt.expr)

    def _visit_ListComprehension(self, stmt):

        # ...
        self.insert_iterator(stmt.iterator)
        self.insert_iterable(stmt.iterable)
        self.rank += 1
        # ...

        # ...
        if not isinstance(stmt.expr, AppliedUndef):
            return self._visit(stmt.expr)
        # ...

        # treat the expression core
        return self._doit()

    def _doit(self):
        # ...
        accelerator = self.accelerator
        parallel    = self.parallel
        inline      = self.inline
        schedule    = self.schedule
        accel       = _accelerator_registery[accelerator]
        # ...

        # ...
        iterator = self.iterators
        iterable = self.iterables
        expr     = self.core
        rank     = self.rank
        # ...

        # ... update rank if using reduction
        if self.op:
            rank = 0
        # ...

#        print('iterator = ', iterator)
#        print('iterable = ', iterable)
#        print('expr     = ', expr    )

        # ... declare lengths and indices
        lengths   = []
        indices   = []
        d_lengths = {}
        d_indices = {}
        for x,xs in zip(iterator, iterable):
            nx   = Variable('int', 'len_'+x.name)
            i_xs = Variable('int', 'i_'+xs.name)

            lengths       += [nx]
            indices       += [i_xs]

            d_lengths[xs]  = nx
            d_indices[xs]  = i_xs
        # ...

        # ...
        func = self.dependencies[self.core.__class__.__name__]
        # ...

        # ...
        results = []
        d_results = {}
        for res in func.results:
            # TODO check if the name exist or use a random name
            if rank == 0:
                name = res.name

            else:
                name  = '{}s'.format(res.name)

            dtype = res.dtype

            var = Variable( dtype,
                            name,
                            rank=rank,
                            allocatable=res.allocatable,
                            is_stack_array = res.is_stack_array,
                            is_pointer=res.is_pointer,
                            is_target=res.is_target,
                            is_polymorphic=res.is_polymorphic,
                            is_optional=res.is_optional,
    #                            shape=None,
    #                            cls_base=None,
    #                            cls_parameters=None,
                            order=res.order,
                            precision=res.precision)

            results += [var]
            d_results[res] = var
        # ...

        # ... create a 1d index if needed
        multi_indices = None
        if rank == 1:
            multi_indices = [Variable('int', 'i_'+r.name) for r in results]
            if len(multi_indices) == 1:
                multi_indices = multi_indices[0]

#            print('> multi indices = ', multi_indices)
        # ...

        # ... initiale value for results
        inits = []
        for r in results:
            value = _get_default_value(r, op=self.op)

            if rank == 0:
                lhs = r
                inits += [Assign(lhs, value)]

#            else:
#                if multi_indices:
#                    lhs = IndexedBase(r.name)[:]
#
#                else:
#                    shape = []
#                    lhs = IndexedBase(r.name)[*shape]
#
#                inits += [Assign(lhs, value)]
        # ...

        # ... assign lengths
        decs = []
        for xs in iterable:
            nx    = d_lengths[xs]
            decs += [Assign(nx, Len(xs))]
        # ...

        # ... create lhs for storing the result
        lhs = []
        for r in results:
            if rank == 0:
                lhs.append(r)

            else:
                if multi_indices:
                    lhs.append(IndexedBase(r.name)[multi_indices])

                else:
                    lhs.append(IndexedBase(r.name)[indices])

        lhs = Tuple(*lhs)
        if len(lhs) == 1:
            lhs = lhs[0]
        # ...

        # ... call to the function to be mapped
        rhs = FunctionCall(func, func.arguments)
        # ...

        # ... create the core statement
        if self.op is None:
            core_stmt = Assign(lhs, rhs)

        else:
            core_stmt = AugAssign(lhs, self.op, rhs)
        # ...

        # ... create loop
        stmts = []

        if multi_indices:
            if len(indices) == 1:
                stmts += [Assign(multi_indices, indices[0])]

            else:
                if not( len(iterable) in [2] ):
                    raise NotImplementedError('')

                # TODO improve formula
                value = indices[0]
                for ix, nx in zip(indices[1:], lengths[::-1][:-1]):
                    value = nx*value + ix

                stmts += [Assign(multi_indices, value)]

        # add core statement
        stmts += [core_stmt]

        for (x, xs) in zip(iterator, iterable):
            nx    = d_lengths[xs]
            ix    = d_indices[xs]

            stmts = [Assign(x, IndexedBase(xs.name)[ix])] + stmts
            stmts = [For(ix, Range(0, nx), stmts, strict=False)]
        # ...

        # ...
        private = ''
        if accelerator:
            private = indices + iterator

            if multi_indices:
                if isinstance(multi_indices, list):
                    private += multi_indices

                else:
                    private += [multi_indices]

            private = ','.join(i.name for i in private)
            private = ' private({private})'.format(private=private)
        # ...

        # ...
        reduction = ''
        if accelerator and self.op:
            if rank > 0:
                raise NotImplementedError('')

            rs = ','.join(i.name for i in results)
            reduction = ' reduction({op}:{args})'.format(op=self.op, args=rs)
        # ...

        # ...
        if accelerator == 'openmp':
            pattern = 'do schedule({schedule}){private}{reduction}'
            accel_stmt = pattern.format( schedule  = schedule,
                                         private   = private,
                                         reduction = reduction )
            prelude = [AnnotatedComment(accel, accel_stmt)]

            accel_stmt = 'end do nowait'
            epilog  = [AnnotatedComment(accel, accel_stmt)]

            stmts = prelude + stmts + epilog

        elif not accelerator is None:
            raise NotImplementedError('')
        # ...

        # ...
        if parallel:
            prelude = [AnnotatedComment(accel, 'parallel')]
            epilog  = [AnnotatedComment(accel, 'end parallel')]

            stmts = prelude + stmts + epilog
        # ...

        # ... update body
        body = inits + decs + stmts
        # ...

        # ...
        if rank == 0:
            body += [Return(results)]

        else:
            # TODO TO BE REMOVED: problem with comments/pragmas
            if accelerator:
                body += [Pass()]
        # ...

        # ... create arguments with appropriate types
        args = []
        d_args = {}
        for arg in self.variables:
            # get the associated dtype
            x = [x for x,xs in zip(iterator, iterable) if xs.name == arg.name ]

            if len(x) == 1:
                x = x[0]

            else:
                x = arg

            # get the call
            call = self.core

            # compute the position in the call
            i_arg = list(call.args).index(x)
            # get the typed argument from the function def
            fargs = self.dependencies[call.__class__.__name__].arguments
            x = fargs[i_arg]

            if not( arg in iterable ):
                var = x

            else:
                if x.rank > 0:
                    raise NotImplementedError('Expecting argument to be a scalar')

                name  = arg.name

                var = Variable( x.dtype,
                                name,
                                rank=1 + x.rank,
                                allocatable=x.allocatable,
                                is_stack_array = x.is_stack_array,
                                is_pointer=x.is_pointer,
                                is_target=x.is_target,
                                is_polymorphic=x.is_polymorphic,
                                is_optional=x.is_optional,
#                                shape=None,
#                                cls_base=None,
#                                cls_parameters=None,
                                order=x.order,
                                precision=x.precision)

            args += [var]
            d_args[arg] = var
        # ...

        # ... update arguments = args + results
        if rank > 0:
            args += results
        # ...

        # ...
        decorators = {'types':         build_types_decorator(args),
                      'external_call': []}

        tag         = random_string( 6 )
        module_name = 'mod_{}'.format( tag)
        g_name      = 'lambda_{}'.format( tag )

        g = FunctionDef(g_name, args, [], body,
                        decorators=decorators)
        # ...

        # ... print python code
        code = pycode(g)

        prelude  = ''
        if not inline:
            prelude = '{prelude}\n\n{code_dep}'.format(prelude=prelude,
                                                       code_dep=self.dependencies_code)


        code = '{prelude}\n\n{code}'.format(prelude=prelude,
                                            code=code)
        # ...

        # ... TODO
        folder = None
        if folder is None:
            basedir = os.getcwd()
            folder = '__pycache__'
            folder = os.path.join( basedir, folder )

        folder = os.path.abspath( folder )
        mkdir_p(folder)
        # ...

        # ...
#        print(code)
#        import sys; sys.exit(0)
        write_code('{}.py'.format(module_name), code, folder=folder)
        # ...

        # ...
        sys.path.append(folder)
        package = importlib.import_module( module_name )
        sys.path.remove(folder)
        # ...

        # we return a module, that will processed by epyccel
        if self.dependencies:
            return package, g_name

        else:
            return getattr(package, g_name)

#==============================================================================
def _lambdify(func, **kwargs):

    if not isinstance(func, FunctionType):
        raise TypeError('Expecting a lambda function')

    # ... get optional arguments
    _kwargs = kwargs.copy()

    namespace = _kwargs.pop('namespace', globals())
    folder    = _kwargs.pop('folder', None)
    # ...

    # ... get the function source code
    func_code = get_source_function(func)
#    print(func_code)
    # ...

    # ...
    tag = random_string( 6 )
    # ...

    # ...
    module_name = 'mod_{}'.format(tag)
    filename    = '{}.py'.format(module_name)
    binary      = '{}.o'.format(module_name)
    # ...

    # ...
    if folder is None:
        basedir = os.getcwd()
        folder = '__pycache__'
        folder = os.path.join( basedir, folder )

    folder = os.path.abspath( folder )
    mkdir_p(folder)
    # ...

    # ...
    write_code(filename, func_code, folder=folder)
    # ...

    # ...
    basedir = os.getcwd()
    os.chdir(folder)
    curdir = os.getcwd()
    # ...

    # ...
    pyccel = Parser(filename, output_folder=folder.replace('/','.'))
    ast = pyccel.parse()

    # TODO shall we keep the annotation here?
    settings = {}
    ast = pyccel.annotate(**settings)

    ns = ast.namespace.symbolic_functions
    if not( len(ns.values()) == 1 ):
        raise ValueError('Expecting one single lambda function')

    func_name = list(ns.keys())[0]
    func      = list(ns.values())[0]
    # ...

    # ...
    if not isinstance(func, Lambda):
        msg = 'Expecting a lambda expr'.format(func_name)
        raise TypeError(msg)
    # ...

    # ... dependencies will contain all the user functions defined functions,
    #     that are needed to lambbdify our expression
    dependencies = {}
    # ...

    # ... annotate functions appearing in the lambda expression
#    print(func.expr)
#    import sys; sys.exit(0)
    calls = list(func.expr.atoms(AppliedUndef))
    calls = [i for i in calls if not( i.__class__.__name__ in _known_functions.keys() )]
    for call in calls:
        # rather than using call.func, we will take the name of the
        # class which defines its type and then the name of the function
        f_name = call.__class__.__name__

        if f_name in namespace.keys():
            f = namespace[f_name]
            dependencies[f_name] = f

        elif not( f_name in _known_functions.keys() ):
            raise ValueError('Unkown function {}'.format(f_name))
    # ...

    # TODO be carefull with the order of dependecies.
    #      => must be corrected in the Parser

    # ... generate ast for dependencies
    code_dep = ''
    code_dep += '\nfrom pyccel.decorators import types'
    code_dep += '\nfrom pyccel.decorators import pure'
    code_dep += '\nfrom pyccel.decorators import external, external_call'

    for f in dependencies.values():
        code_dep = '{code}\n\n{new}'.format( code = code_dep,
                                             new  = get_source_function(f) )

    write_code(filename, code_dep, folder=folder)

    pyccel = Parser(filename, output_folder=folder.replace('/','.'))
    ast = pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)
    dependencies = ast.namespace.functions
    # ...

    # ...
    visitor = VisitorLambda( func,
                             dependencies=dependencies,
                             dependencies_code=code_dep,
                             **kwargs)
    # ...

    return visitor._visit(visitor.expr)
