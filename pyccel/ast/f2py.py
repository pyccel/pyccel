# -*- coding: utf-8 -*-

from sympy import Tuple

from pyccel.ast.core import FunctionCall
from pyccel.ast.core import FunctionDef, F2PYFunctionDef
from pyccel.ast.core import Variable
from pyccel.ast.core import ValuedVariable
from pyccel.ast.core import Assign
from pyccel.ast.core import Import
from pyccel.ast.core import AsName
from pyccel.ast.core import Comment

from pyccel.ast.datatypes import str_dtype

__all__ = (
   'as_static_function',
   'as_static_function_call',
   'sanitize_arguments',
)

#=======================================================================================
def sanitize_arguments(args):
    _args = []
    for a in args:
        if isinstance( a, Variable ):
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

    # Convert array results to inout arguments
    for r in results:
        if r.rank > 0 and r not in args:
            args += [r]
            arguments_inout += [False]
        elif r.rank == 0:
            _results += [r]

    if name is None:
        name = 'f2py_{}'.format(func.name).lower()

    # ...
    results_names = [i.name for i in _results]
    _args = []
    _arguments_inout = []

    for i_a, a in enumerate(args):
        if not isinstance(a, Variable):
            raise TypeError('Expecting a Variable type for {}'.format(a))

        rank = a.rank
        if rank > 0:
            # ...
            additional_args = []
            for i in range(rank):
                n_name = 'n{i}_{name}'.format(name=str(a.name), i=i)
                n_arg  = Variable('int', n_name, precision=4)

                additional_args += [n_arg]

            shape_new = Tuple(*additional_args, sympify=False)
            # ...

            _args += additional_args
            for j in additional_args:
                _arguments_inout += [False]

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

    # ... If needed, add !f2py instructions at beginning of function body
    f2py_template = 'f2py integer(kind={kind}) :: n{index}_{name}=shape({name},{index_t})'
    f2py_instructions = []
    int_kind = Variable('int', 'f2py_array_dimension').precision

    for a in args:
        if a.rank > 1 and a.order == 'C':

            # Reverse shape of array
            transpose_stmts = [Comment(f2py_template.format(kind    = int_kind,
                                                            index   = i,
                                                            name    = a.name,
                                                            index_t = str(a.rank - i - 1)))
                               for i in range(a.rank)]

            # Tell f2py that array has C ordering
            c_intent_stmt = Comment('f2py intent(c) {}'.format(a.name))

            # Update f2py directives
            f2py_instructions += [*transpose_stmts, c_intent_stmt]
        elif isinstance(a,ValuedVariable):
            f2py_instructions += [Comment('f2py {type} :: {name} = {default_value}'.format(type = str_dtype(a.dtype),
                name = a.name, default_value = a.value))]

    if f2py_instructions:
        body = f2py_instructions + body.body
    # ...

    return F2PYFunctionDef( name, list(args), results, body,
                        local_vars = func.local_vars,
                        is_static = True,
                        arguments_inout = arguments_inout,
                        functions = functions,
                        imports = func.imports
                        )

#=======================================================================================
def as_static_function_call(func, mod_name, name=None):

    assert isinstance(func, FunctionDef)
    assert isinstance(mod_name, str)

    # create function alias by prepending 'mod_' to its name
    func_alias = func.rename('mod_' + str(func.name))

    # from module import func as func_alias
    imports = [Import(target=AsName(func_alias.name, func.name), source=mod_name)]

    # function arguments
    args = sanitize_arguments(func.arguments)
    # function body
    call = FunctionCall(func_alias, args)
    stmt = call if func.is_procedure else Assign(func.results[0], call)
    body = [stmt]

    # new function declaration
    new_func = FunctionDef(func.name, list(args), func.results, body,
                       arguments_inout = func.arguments_inout,
                       functions = func.functions,
                       imports = imports,
                       )

    # make it compatible with f2py
    static_func = as_static_function(new_func, name)

    return static_func
