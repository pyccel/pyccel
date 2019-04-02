# -*- coding: UTF-8 -*-

import os
import sys
import importlib
import numpy as np
from types import FunctionType

from sympy import Indexed, IndexedBase, Tuple, Lambda
from sympy.core.function import AppliedUndef
from sympy.core.function import UndefinedFunction
from sympy import Sum

from pyccel.codegen.utilities       import construct_flags as construct_flags_pyccel
from pyccel.codegen.utilities       import execute_pyccel
from pyccel.epyccel import get_source_function
from pyccel.epyccel import random_string
from pyccel.epyccel import write_code
from pyccel.epyccel import mkdir_p
from pyccel.epyccel import get_function_from_ast
from pyccel.ast.datatypes import dtype_and_precsision_registry as dtype_registry
from pyccel.ast import Variable, Len, Assign, AugAssign
from pyccel.ast import For, Range, FunctionDef
from pyccel.ast import FunctionCall
from pyccel.ast import Comment, AnnotatedComment
from pyccel.ast import Print, Pass, Return
from pyccel.ast import ListComprehension
from pyccel.ast.datatypes import NativeInteger
from pyccel.codegen.printing.pycode import pycode
from pyccel.codegen.printing.fcode  import fcode
from pyccel.ast.utilities import build_types_decorator
from pyccel.parser import Parser

_avail_patterns = ['map']

#==============================================================================
_accelerator_registery = {'openmp': 'omp', 'openacc': 'acc', None: None}
_known_functions_registery = {'sum': '+'}

#==============================================================================
def _extract_core_expr(expr):
    """extract core expression from a lambda expression"""
    if isinstance(expr, Lambda):
        return _extract_core_expr(expr.expr)

    elif isinstance(expr, ListComprehension):
        return _extract_core_expr(expr.expr)

    elif isinstance(expr, AppliedUndef):
        name = expr.__class__.__name__
        if name in _known_functions_registery.keys():
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
        self._op = _known_functions_registery[op]

    def _visit(self, stmt):

        cls = type(stmt)
        name = cls.__name__

        method = '_visit_{}'.format(name)
        if hasattr(self, method):
            return getattr(self, method)(stmt)

        elif name in _known_functions_registery.keys():
            self._set_op(name)
            # TODO must reset op
            args = stmt.args
            if name == 'sum':
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
            rank -= 1
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
            value = None
            if isinstance(r.dtype, NativeInteger):
                if self.op == '*':
                    value = 1

                else:
                    value = 0

            else:
                raise NotImplementedError('')

            lhs = r
            if rank > 0:
                raise NotImplementedError('')

            inits += [Assign(lhs, value)]
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
def lambdify(pattern, *args, **kwargs):

    if isinstance(pattern, FunctionType):
        assert(len(args) == 0)
        return _lambdify_func(pattern, **kwargs)

    if not isinstance(pattern, str):
        raise TypeError('Expecting a string for pattern')

    if not pattern in _avail_patterns:
        raise ValueError('No pattern {} found'.format(pattern))

    _lambdify = eval('_lambdify_{}'.format(pattern))

    return _lambdify(*args, **kwargs)

#==============================================================================
def _lambdify_func(func, **kwargs):

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
    calls = func.expr.atoms(AppliedUndef)
    for call in calls:
        # rather than using call.func, we will take the name of the
        # class which defines its type and then the name of the function
        f_name = call.__class__.__name__

        if f_name in namespace.keys():
            f = namespace[f_name]
            dependencies[f_name] = f

        elif not( f_name in _known_functions_registery.keys() ):
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


#==============================================================================
def _lambdify_map(*args, **kwargs):

    # ... get arguments
    func, ranks = _extract_args_map(*args, **kwargs)
    # ...

    # ... get optional arguments
    _kwargs = kwargs.copy()

    namespace         = _kwargs.pop('namespace'        , globals())
    compiler          = _kwargs.pop('compiler'         , 'gfortran')
    fflags            = _kwargs.pop('fflags'           , None)
    accelerator       = _kwargs.pop('accelerator'      , None)
    verbose           = _kwargs.pop('verbose'          , False)
    debug             = _kwargs.pop('debug'            , False)
    include           = _kwargs.pop('include'          , [])
    libdir            = _kwargs.pop('libdir'           , [])
    modules           = _kwargs.pop('modules'          , [])
    libs              = _kwargs.pop('libs'             , [])
    extra_args        = _kwargs.pop('extra_args'       , '')
    folder            = _kwargs.pop('folder'           , None)
    mpi               = _kwargs.pop('mpi'              , False)
    assert_contiguous = _kwargs.pop('assert_contiguous', False)

    # TODO
    accel    = None
    schedule = None
    if accelerator is None:
        accelerator = 'openmp'
        accel       = 'omp'
        schedule    = 'static'
#        schedule    = 'runtime'

    if fflags is None:
        fflags = construct_flags_pyccel( compiler,
                                         fflags=None,
                                         debug=debug,
                                         accelerator=accelerator,
                                         include=[],
                                         libdir=[] )
    # ...

    # ... parallel options
    parallel = _kwargs.pop('parallel', True)
    # ...

    # ... additional options
    inline = _kwargs.pop('inline', False)
    # ...

    # ... get the function source code
    func_code = get_source_function(func)
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
    filename, ast = execute_pyccel( filename,
                                    compiler     = compiler,
                                    fflags       = fflags,
                                    debug        = debug,
                                    verbose      = verbose,
                                    accelerator  = accelerator,
                                    modules      = modules,
                                    convert_only = True,
                                    return_ast   = True )
    # ...

    # ... construct a f2py interface for the assembly
    # be careful: because of f2py we must use lower case
    func_name = func.__name__
    funcs     = ast.routines + ast.interfaces
    func      = get_function_from_ast(funcs, func_name)
    namespace = ast.parser.namespace.sons_scopes
    # ...

    # ...
    if not(len(func.arguments) == len(ranks)):
        raise ValueError('Wrong number of ranks for function arguments')
    # ...

#    # ...
#    print(func.arguments)
#    print([i.dtype for i in func.arguments])
#    print(ranks)
#    # ...

    # ...
    extended_args = [i for i,rank in zip(func.arguments, ranks) if rank > 0]
    # ...

    # ...
    args = []
    d_args = {}
    for arg,rank in zip(func.arguments, ranks):
        if (rank == 0) or (rank is None):
            var = arg

        else:
            name  = 'arr_{}'.format(arg.name)
            dtype = arg.dtype

            if arg.rank > 0:
                raise ValueError('Expecting argument to be a scalar')

            var = Variable( dtype,
                            name,
                            rank=rank,
                            allocatable=arg.allocatable,
                            is_stack_array = arg.is_stack_array,
                            is_pointer=arg.is_pointer,
                            is_target=arg.is_target,
                            is_polymorphic=arg.is_polymorphic,
                            is_optional=arg.is_optional,
#                            shape=None,
#                            cls_base=None,
#                            cls_parameters=None,
                            order=arg.order,
                            precision=arg.precision)

        args += [var]
        d_args[arg] = var
    # ...

#    print(args)

    # ... declare lengths and indices
    lengths   = []
    d_lengths = {}
    indices   = []
    d_indices = {}
    for arg in extended_args:
        x     = d_args[arg]
        nx    = Variable('int', 'len_'+x.name)
        i_arg = Variable('int', 'i_'+arg.name)

        lengths      += [nx]
        d_lengths[x]  = nx

        indices      += [i_arg]
        d_indices[x]  = i_arg
    # ...

    # ...
    results = []
    d_results = {}
    for res in func.results:
        name  = 'arr_{}'.format(res.name)
        dtype = res.dtype

        var = Variable( dtype,
                        name,
                        rank=int(np.asarray(ranks).sum()),
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

    # ... declare an empty body
    body = []
    # ...

    # ... assign lengths
    for arg in extended_args:
        x  = d_args[arg]
        nx = d_lengths[x]
        stmt = Assign(nx, Len(x))

        body += [stmt]
    # ...

    # ... call to the function to be mapped
    lhs = []
    for r in results:
        lhs.append(IndexedBase(r.name)[indices])

    lhs = Tuple(*lhs)
    if len(lhs) == 1:
        lhs = lhs[0]

    rhs = FunctionCall(func, func.arguments)

    stmts = [Assign(lhs, rhs)]
    # ...

    # ... create loop
    for x in extended_args:
        arr_x = d_args[x]
        nx    = d_lengths[arr_x]
        ix    = d_indices[arr_x]

        stmts = [Assign(x, IndexedBase(arr_x.name)[ix])] + stmts
        stmts = [For(ix, Range(0, nx), stmts, strict=False)]
    # ...

    # ...
    private = ''
    if accelerator:
        private = indices + extended_args
        private = ','.join(i.name for i in private)
        private = 'private({private})'.format(private=private)
    # ...

    # ...
    if accelerator == 'openmp':
        accel_stmt = 'do schedule({schedule}) {private}'.format(schedule=schedule,
                                                                private=private)
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
    body += stmts
    # ...

    # ... TODO TO BE REMOVED: problem with comments/pragmas
#    body += [Print(['Done'])]
    body += [Pass()]
    # ...

    # ... update arguments = args + results
    args += results
    # ...

    # ...
    decorators = {'types':         build_types_decorator(args),
                  'external_call': []}

    g_name = 'map_{}'.format(func_name)
    g = FunctionDef(g_name, args, [], body,
                    decorators=decorators)
    # ...

    # ... print python code
    code = pycode(g)

    prelude  = ''
    prelude += '\nfrom pyccel.decorators import types'
    prelude += '\nfrom pyccel.decorators import pure'
    prelude += '\nfrom pyccel.decorators import external, external_call'

    if not inline:
        prelude = '{prelude}\n\n{func_code}'.format(prelude=prelude,
                                                    func_code=func_code)


    code = '{prelude}\n\n{code}'.format(prelude=prelude,
                                        code=code)
    # ...

    print(code)

    # ...
    write_code('{}.py'.format(module_name), code, folder=folder)
    # ...

    # ...
    sys.path.append(folder)
    package = importlib.import_module( module_name )
    sys.path.remove(folder)
    # ...

    return package

#    return getattr(package, g_name)

#==============================================================================
# TODO allow rank to be a dictionary
def _extract_args_map(*args, **kwargs):
    assert(len(args) == 1)

    func = args[0]
    if not isinstance(func, FunctionType):
        raise TypeError('> Expecting a function')

    rank = kwargs.pop('rank', None)
    if rank is None:
        raise ValueError('rank must be provided as optional argument')

    return func, rank
