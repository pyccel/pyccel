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
from pyccel.ast.core import Slice, String
from pyccel.ast import Zeros
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
# TODO move as method of FunctionDef
def get_results_shape(func):
    """returns a dictionary that contains for each result, its shape. When using
    the decorator @shapes, the shape value may be computed"""

    # ...
    arguments       = list(func.arguments)
    arguments_inout = list(func.arguments_inout)
    results         = list(func.results)

    inouts = [x for x,flag in zip(arguments, arguments_inout) if flag]
    # ...

    # ...
    d_args = {}
    for a in arguments:
        d_args[a.name] = a
    # ...

#    print('results = ', results)
#    print('inouts   = ', inouts)

    d_shapes = {}
    if 'shapes' in func.decorators.keys():
        d = func.decorators['shapes']
        for valued in d:
            # ...
            r = [r for r in results + inouts if r.name == valued.name]
            if not r:
                raise ValueError('Could not find {}'.format(r))

            assert(len(r) == 1)
            r = r[0]
            # ...

            # ...
            rhs = valued.value
            if isinstance(rhs, String):
                rhs = rhs.arg.replace("'",'')

            else:
                raise NotImplementedError('')
            # ...

            # ...
            if rhs in d_args.keys():
                rhs = d_args[str(rhs)]

            else:
                raise NotImplementedError('TODO {}'.format(type(rhs)))
            # ...

            # TODO improve
            # this must always be a list of slices
            d_shapes[r.name] = [Slice(None, rhs)]

    # TODO treate the case when shapes is not given => add some checks
#    else:
#        raise NotImplementedError('')

    return d_shapes


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

        # ... TODO improve
        #     define inouts variables, that are local to the lambda expression
        inouts = [x for x,flag in zip(func.arguments, func.arguments_inout) if flag]
        # ...

        # ... workplace contains variables that are defined localy in the lambda
        #     expression
        local_names = []
        local_vars  = []
        if 'workplace' in func.decorators.keys():
            local_names = [x.arg.replace("'","") for x in func.decorators['workplace']]
            local_vars  = [x     for x in func.arguments if x.name in local_names]
        # ...

        # ... construct out variables
        #     when using a reduction operator and having results of rank > 0,
        #     then we must add new variables for the reduced value,
        #     in this case, out_var will be defined by the reduced variables,
        #     while the previous out_vars will be appended to the list of
        #     local_vars
        out_vars  = [r for r in inouts+list(func.results) if not( r.name in local_names )]

        d_local_reduced = {}
        if self.op:
            local_vars  += [i      for i in out_vars if i.rank > 0]
            local_names += [i.name for i in out_vars if i.rank > 0]

            _out_vars = []
            for res in out_vars:

                if rank == 0:
                    name = 'reduced_{}'.format(res.name)

                else:
                    raise NotImplementedError('')

                dtype = res.dtype

                var = Variable( dtype,
                                name,
                                rank=res.rank,
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

                _out_vars += [var]
                d_local_reduced[res] = var

            out_vars = _out_vars

        out_names = [r.name for r in out_vars]
        # ...

#        print('>>> out_vars   = ', out_vars)
#        print('>>> local_vars = ', local_vars)
#        print('>>> d_local_reduced = ', d_local_reduced)
#        import sys; sys.exit(0)

        # ... get shape for results and local variables
        d_shapes = get_results_shape(func)
        # ...

        # ...
        results = []
        d_results = {}
        for res in out_vars:
            # TODO check if the name exist or use a random name
            if rank == 0:
                name = res.name

            else:
                name  = '{}s'.format(res.name)

            dtype = res.dtype

            var = Variable( dtype,
                            name,
                            rank=res.rank+rank,
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

        # ... create indices for local reduced variables
        d_reduced_indices = {}
        for var, reduced_var in d_local_reduced.items():
            names = ['i{d}_{name}'.format(d=d, name=reduced_var.name) for d in range(var.rank)]
            _indices = [Variable('int', name) for name in names]

            d_reduced_indices[var] = _indices
        # ...

        # ... allocate local variables
        allocations = []
        d_shapes_values = {}
        for x in local_vars:
            shape = []
            for _slice in d_shapes[x.name]:
                start = _slice.start
                end   = _slice.end
                if start is None:
                    shape.append(end)

                else:
                    raise NotImplementedError('')

            # must be done here, since we need it as a list later
            d_shapes_values[x.name] = shape

            if len(shape) == 1:
                shape = shape[0]

            else:
                raise NotImplementedError('')

            # TODO improve
            dtype = str(x.dtype)
            allocations += [Assign(x, Zeros(shape, dtype))]
        # ...

        # ... initiale value for results
        inits = []
        for r in results:
            value = _get_default_value(r, op=self.op)

            if rank == 0:
                lhs = r
                inits += [Assign(lhs, value)]

            else:
                ls = [Slice(None, None) for i in range(0, r.rank)]
                lhs = IndexedBase(r.name)[ls]

                inits += [Assign(lhs, value)]
        # ...

        # ... assign lengths
        decs = []
        for xs in iterable:
            nx    = d_lengths[xs]
            decs += [Assign(nx, Len(xs))]
        # ...

        # ...
        ind = indices
        if multi_indices:
            ind = multi_indices
        # ...

        # ... create lhs for storing the result
        lhs = []
        for r in results:
            if rank == 0:
                lhs.append(r)

            else:
                lhs.append(IndexedBase(r.name)[ind])

        lhs = Tuple(*lhs)
        if len(lhs) == 1:
            lhs = lhs[0]
        # ...

        # ... build call arguments
        arguments = []
        for x in func.arguments:
            arg = x
            if x.name in out_names:
                xs = d_results[x]

                ls = []
                # add ':' depending on the rank of the result
                for i in range(0, x.rank):
                    ls += [Slice(None, None)]

                if rank > 0:
                    # TODO WHICH ORDER TO CHOOSE?
                    if isinstance(ind, Variable):
                        ls = [ind] + ls

                    else:
                        ls = ind + ls

                arg = IndexedBase(xs.name)[ls]

            arguments += [arg]
        # ...

        # ... call to the function to be mapped
        rhs = FunctionCall(func, arguments)
        # ...

        # ... create the core statement
        if len(func.results) == 0:
            core_stmt = rhs

        else:
            if self.op is None:
                core_stmt = Assign(lhs, rhs)

            else:
                core_stmt = AugAssign(lhs, self.op, rhs)

        core_stmts = [core_stmt]
        # ...

        # ... when using a reduction operator on a local variable,
        #     we add the local reduction to the core stmt
        reduction_stmts = []
        for var, reduced_var in d_local_reduced.items():
            if var.rank > 0:
                _indices = d_reduced_indices[var]
                _stmts = [AugAssign(reduced_var, self.op, IndexedBase(var.name)[_indices])]

                shape = d_shapes_values[var.name]
                if not isinstance(shape, (list, tuple, Tuple)):
                    shape = [shape]

                # TODO improve
                for i, n in zip(_indices, shape):
                    _stmts = [For(i, Range(0, n), _stmts, strict=False)]

                reduction_stmts += _stmts

        core_stmts += reduction_stmts
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
        stmts += core_stmts

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

            # add local variables
            private += local_vars

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
        body = allocations + inits + decs + stmts
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

    namespace       = _kwargs.pop('namespace', globals())
    folder          = _kwargs.pop('folder', None)
    functional_args = _kwargs.pop('functional_args', None)
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
    # TODO have a function for these imports
    code_dep = ''
    code_dep += '\nfrom pyccel.decorators import types'
    code_dep += '\nfrom pyccel.decorators import pure'
    code_dep += '\nfrom pyccel.decorators import external, external_call'
    code_dep += '\nfrom pyccel.decorators import shapes'
    code_dep += '\nfrom pyccel.decorators import workplace'

    # TODO improve
    code_dep += '\nfrom numpy import zeros'
    code_dep += '\nfrom numpy import float64'

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
