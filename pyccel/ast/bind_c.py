# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from pyccel.ast.core import FunctionCall, Module
from pyccel.ast.core import FunctionAddress
from pyccel.ast.core import FunctionDef, BindCFunctionDef
from pyccel.ast.core import Assign
from pyccel.ast.core import Import
from pyccel.ast.core import AsName
from pyccel.ast.variable import Variable
from pyccel.parser.scope import Scope

__all__ = (
    'as_static_module',
    'as_static_function',
    'as_static_function_call',
    'sanitize_arguments',
)

#=======================================================================================
def sanitize_arguments(args):
    _args = []
    for a in args:
        if isinstance(a.var, (Variable, FunctionAddress)):
            _args.append(a.var)

        else:
            raise NotImplementedError('TODO for {}'.format(type(a)))

    return _args

#=======================================================================================
def as_static_function(func, *, mod_scope, name=None):
    """ Translate a FunctionDef to a BindCFunctionDef by altering the
    arguments to allow the function to be called from c.
    E.g. the size of each dimension of an array is provided

    Parameters
    ==========
    func     : FunctionDef
               The function to be translated
    mod_scope: Scope
               The scope of the module which contains func
    name     : str
               The new name of the function
    """

    assert(isinstance(func, FunctionDef))

    args    = list(func.arguments)
    results = list(func.results)
    body    = func.body
    arguments_inout = func.arguments_inout
    functions = func.functions
    _results = []
    interfaces = func.interfaces

    scope = mod_scope.new_child_scope(func.name)

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
        a = a.var
        if not isinstance(a, (Variable, FunctionAddress)):
            raise TypeError('Expecting a Variable or FunctionAddress type for {}'.format(a))
        if not isinstance(a, FunctionAddress) and a.rank > 0:
            # ...
            additional_args = []
            for i in range(a.rank):
                n_name = 'n{i}_{name}'.format(name=a.name, i=i)
                n_arg  = Variable('int', n_name)

                additional_args += [n_arg]

            shape_new = tuple(additional_args)
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
                        is_static = True,
                        arguments_inout = arguments_inout,
                        functions = functions,
                        interfaces = interfaces,
                        imports = func.imports,
                        original_function = func,
                        doc_string = func.doc_string,
                        scope = scope
                        )

#=======================================================================================
def as_static_module(funcs, original_module):
    """ Create the module contained in the bind_c_mod.f90 file
    This is the interface between the c code and the fortran code thanks
    to iso_c_bindings

    Parameters
    ==========
    funcs : list of FunctionDef
            All the functions which may be exposed to c
    original_module : Module
            The module being wrapped
    """
    funcs = [f for f in funcs if not f.is_private]
    imports = []
    scope = Scope(used_symbols = original_module.scope.local_used_symbols.copy())
    bind_c_funcs = [as_static_function_call(f, original_module, scope, imports = imports) for f in funcs]
    if isinstance(original_module.name, AsName):
        name = scope.get_new_name('bind_c_{}'.format(original_module.name.target))
    else:
        name = scope.get_new_name('bind_c_{}'.format(original_module.name))
    return Module(name, (), bind_c_funcs, imports = imports, scope=scope)

#=======================================================================================
def as_static_function_call(func, mod, mod_scope, name=None, imports = None):
    """ Translate a FunctionDef to a BindCFunctionDef which calls the
    original function. A BindCFunctionDef is a FunctionDef where the
    arguments are altered to allow the function to be called from c.
    E.g. the size of each dimension of an array is provided

    Parameters
    ==========
    func     : FunctionDef
               The function to be translated
    mod      : Module
               The module which contains func
    name     : str
               The new name of the function
    imports  : list
               An optional parameter into which any required imports
               can be collected
    """

    assert isinstance(func, FunctionDef)
    assert isinstance(mod, Module)
    mod_name = mod.scope.get_python_name(mod.name)

    # from module import func
    if imports is None:
        local_imports = [Import(target=AsName(func, func.name), source=mod_name, mod=mod)]
    else:
        imports.append(Import(target=AsName(func, func.name), source=mod_name, mod=mod))
        local_imports = ()

    # function arguments
    args = sanitize_arguments(func.arguments)
    # function body
    call    = FunctionCall(func, args)
    results = func.results
    results = results[0] if len(results) == 1 else results
    stmt    = call if len(func.results) == 0 else Assign(results, call)
    body    = [stmt]

    # new function declaration
    new_func = FunctionDef(func.name, func.arguments, func.results, body,
                       arguments_inout = func.arguments_inout,
                       functions = func.functions,
                       interfaces = func.interfaces,
                       imports = local_imports,
                       doc_string = func.doc_string
                       )

    # make it compatible with c
    static_func = as_static_function(new_func, name=name, mod_scope=mod_scope)

    return static_func
