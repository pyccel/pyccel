# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from pyccel.ast.basic import Basic
from pyccel.ast.core import CodeBlock, FunctionCall, Module
from pyccel.ast.core import FunctionAddress
from pyccel.ast.core import FunctionDef
from pyccel.ast.core import Assign
from pyccel.ast.core import Import
from pyccel.ast.core import AsName
from pyccel.ast.core import Allocate
from pyccel.ast.datatypes import DataType, NativeInteger
from pyccel.ast.variable import Variable
from pyccel.parser.scope import Scope

__all__ = (
    'BindCFunctionDef',
    'BindCPointer',
    'CLocFunc',
    'as_static_module',
    'as_static_function',
    'as_static_function_call',
    'sanitize_arguments',
    'wrap_array',
    'wrap_module_array_var',
)

class BindCFunctionDef(FunctionDef):
    """
    Contains the c-compatible version of the function which is
    used for the wrapper.
    As compared to a normal FunctionDef, this version contains
    arguments for the shape of arrays. It should be generated by
    calling ast.bind_c.as_static_function

    Parameters
    ----------
    *args : See FunctionDef

    original_function : FunctionDef
        The function from which the c-compatible version was created
    """
    __slots__ = ('_original_function',)
    _attribute_nodes = (*FunctionDef._attribute_nodes, '_original_function')

    def __init__(self, *args, original_function, **kwargs):
        self._original_function = original_function
        super().__init__(*args, **kwargs)
        assert self.name == self.name.lower()

    @property
    def original_function(self):
        """ The function which is wrapped by this BindCFunctionDef
        """
        return self._original_function

#=======================================================================================
def sanitize_arguments(args):
    """ Ensure that all arguments are of expected types (Variable or FunctionAddress)
    """
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
            #wrap the array that is returned from the original function
            array_body, array_vars = wrap_array(r, scope, False)
            scope.insert_variable(array_vars[-1])
            scope.insert_variable(r)
            body = CodeBlock(func.body.body + tuple(array_body))
            array_vars.pop(-1)
            _results += array_vars
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
                              memory_handling = a.memory_handling,
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
    variables = [f for f in original_module.variables if not f.is_private]
    imports = []
    scope = Scope(used_symbols = original_module.scope.local_used_symbols.copy())
    bind_c_funcs = [as_static_function_call(f, original_module, scope, imports = imports) for f in funcs]
    bind_c_arrays = [wrap_module_array_var(v, scope, original_module) for v in variables if v.rank > 0]
    if isinstance(original_module.name, AsName):
        name = scope.get_new_name('bind_c_{}'.format(original_module.name.target))
    else:
        name = scope.get_new_name('bind_c_{}'.format(original_module.name))
    return Module(name, (), bind_c_funcs+bind_c_arrays, imports = imports, scope=scope)

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

#=======================================================================================

class BindCPointer(DataType):
    """ Datatype representing a c pointer in fortran
    """
    __slots__ = ()
    _name = 'bindcpointer'

#=======================================================================================

class CLocFunc(Basic):
    """ Class representing the iso_c_binding function cloc which returns a valid
    C pointer to the location where an object can be found
    """
    __slots__ = ('_arg', '_result')
    _attribute_nodes = ()
    def __init__(self, argument, result):
        self._arg = argument
        self._result = result
        super().__init__()

    @property
    def arg(self):
        """ Object which will be pointed at
        """
        return self._arg

    @property
    def result(self):
        """ The variable in which the pointer is stored
        """
        return self._result

#=======================================================================================

def wrap_array(var, scope, persistent):
    """ Function returning the code and local variables necessary to wrap an array

    Parameters
    ----------
    var : Variable
            The array to be wrapped
    scope : Scope
            The current scope (used to find valid names for variables)
    persistent : bool
            Indicates whether the variable is persistent in memory or
            if it needs copying to avoid dead pointers

    Results
    -------
    body : list
            A list describing the lines which must be printed to wrap the array
    variables : list
            A list of all new variables necessary to wrap the array. The list
            contains:
            - The C Pointer which wraps the array
            - Variables containing the sizes of the array
            - The Fortran pointer which will contain a copy of the Fortran data
              (unless the variable is persistent in memory)
    """
    bind_var = Variable(dtype=BindCPointer(), name = scope.get_new_name('bound_'+var.name))
    sizes = [Variable(dtype=NativeInteger(), name = scope.get_new_name()) for _ in range(var.rank)]
    assigns = [Assign(sizes[i], var.shape[i]) for i in range(var.rank)]
    variables = [bind_var, *sizes]
    if not persistent:
        ptr_var = var.clone(scope.get_new_name(var.name+'_ptr'),
                memory_handling='alias')
        alloc = Allocate(ptr_var, shape=var.shape, order=var.order, status='unallocated')
        copy  = Assign(ptr_var, var)
        c_loc = CLocFunc(ptr_var, bind_var)
        variables.append(ptr_var)
        body = [*assigns, alloc, copy, c_loc]
    else:
        c_loc = CLocFunc(var, bind_var)
        body = [*assigns, c_loc]
    return body, variables

#=======================================================================================

def wrap_module_array_var(var, scope, mod):
    """ Function returning the function necessary to expose an array

    Parameters
    ----------
    var : Variable
            The array to be exposed
    scope : Scope
            The current scope (used to find valid names for variables)
    mod : Module
            The module where the variable is defined

    Results
    -------
    func : FunctionDef
            A function which wraps an array and can be called from C
    """
    func_name = 'bind_c_'+var.name
    func_scope = scope.new_child_scope(func_name)
    body, necessary_vars = wrap_array(var, func_scope, True)
    func_scope.insert_variable(necessary_vars[0])
    arg_vars = necessary_vars
    import_mod = Import(mod.name, AsName(var,var.name), mod=mod)
    func = BindCFunctionDef(name = func_name,
                  body      = body,
                  arguments = [],
                  results   = arg_vars,
                  imports   = [import_mod],
                  scope = func_scope,
                  original_function = None)
    return func
