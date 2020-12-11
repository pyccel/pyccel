# -*- coding: utf-8 -*-

from sympy import Tuple

from pyccel.ast.core import FunctionCall
from pyccel.ast.core import FunctionAddress
from pyccel.ast.core import FunctionDef, BindCFunctionDef
from pyccel.ast.core import Variable
from pyccel.ast.core import Assign
from pyccel.ast.core import Import
from pyccel.ast.core import AsName
from pyccel.ast.core import IndexedVariable

__all__ = (
   'as_static_function',
   'as_static_function_call',
   'sanitize_arguments',
)

#=======================================================================================
def sanitize_arguments(args):
    _args = []
    for a in args:
        if isinstance(a, (Variable, FunctionAddress)):
            _args.append(a)

        elif isinstance( a, IndexedVariable ):
            a_new = Variable( a.dtype, str(a.name),
                              shape       = a.shape,
                              rank        = a.rank,
                              order       = a.order,
                              precision   = a.precision)

            _args.append(a_new)

        else:
            raise NotImplementedError('TODO for {}'.format(type(a)))

    return _args

#=======================================================================================
def as_static_function(func, name=None):

    assert(isinstance(func, FunctionDef))

    args    = list(func.arguments)
    results = list(func.results)
    body    = func.body
    arguments_inout = func.arguments_inout
    functions = func.functions
    _results = []
    interfaces = func.interfaces

    # Convert array results to inout arguments
    for r in results:
        if r.rank > 0 and r not in args:
            args += [r]
            arguments_inout += [False]
        elif r.rank == 0:
            _results += [r]

    if name is None:
        name = 'bind_c_{}'.format(func.name).lower()

    # ...
    results_names = [i.name for i in _results]
    _args = []
    _arguments_inout = []

    for i_a, a in enumerate(args):
        if not isinstance(a, (Variable, FunctionAddress)):
            raise TypeError('Expecting a Variable or FunctionAddress type for {}'.format(a))
        if not isinstance(a, FunctionAddress) and a.rank > 0:
            # ...
            additional_args = []
            for i in range(a.rank):
                n_name = 'n{i}_{name}'.format(name=str(a.name), i=i)
                n_arg  = Variable('int', n_name, precision=4)

                additional_args += [n_arg]

            shape_new = Tuple(*additional_args, sympify=False)
            # ...

            _args += additional_args
            _arguments_inout += [False] * len(additional_args)

            a_new = Variable( a.dtype, a.name,
                              allocatable = a.allocatable,
                              is_pointer  = a.is_pointer,
                              is_target   = a.is_target,
                              is_optional = a.is_optional,
                              shape       = shape_new,
                              rank        = a.rank,
                              order       = a.order,
                              precision   = a.precision)

            if not( a.name in results_names ):
                _args += [a_new]

            else:
                _results += [a_new]

        else:
            _args += [a]

        intent = arguments_inout[i_a]
        _arguments_inout += [intent]
    args = _args
    results = _results
    arguments_inout = _arguments_inout
    # ...
    return BindCFunctionDef( name, list(args), results, body,
                        local_vars = func.local_vars,
                        is_static = True,
                        arguments_inout = arguments_inout,
                        functions = functions,
                        interfaces = interfaces,
                        imports = func.imports,
                        original_function = func,
                        doc_string = func.doc_string,
                        )

#=======================================================================================
def as_static_function_call(func, mod_name, name=None):

    assert isinstance(func, FunctionDef)
    assert isinstance(mod_name, str)

    # create function alias by prepending 'mod_' to its name
    func_alias = func.clone('mod_' + str(func.name))

    # from module import func as func_alias
    imports = [Import(target=AsName(func.name, func_alias.name), source=mod_name)]

    # function arguments
    args = sanitize_arguments(func.arguments)
    # function body
    call    = FunctionCall(func_alias, args)
    results = func.results
    results = results[0] if len(results) == 1 else results
    stmt    = call if len(func.results) == 0 else Assign(results, call)
    body    = [stmt]

    # new function declaration
    new_func = FunctionDef(func.name, list(args), func.results, body,
                       arguments_inout = func.arguments_inout,
                       functions = func.functions,
                       interfaces = func.interfaces,
                       imports = imports,
                       doc_string = func.doc_string,
                       )

    # make it compatible with c
    static_func = as_static_function(new_func, name)

    return static_func
