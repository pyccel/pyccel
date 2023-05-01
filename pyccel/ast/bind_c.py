# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
# ------------------------------------------------------------------------------------------#

from pyccel.ast.basic import Basic
from pyccel.ast.core import CodeBlock, FunctionCall, Module
from pyccel.ast.core import FunctionAddress
from pyccel.ast.core import FunctionDef
from pyccel.ast.core import FunctionDefArgument, FunctionDefResult
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
    'as_static_function',
    'as_static_function_call',
    'as_static_module',
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
        assert all(isinstance(a, BindCFunctionDefArgument) for a in self._arguments)
        assert all(isinstance(a, BindCFunctionDefResult) for a in self._results)

    @property
    def original_function(self):
        """ The function which is wrapped by this BindCFunctionDef
        """
        return self._original_function

    @property
    def bind_c_arguments(self):
        """
        Get the BindCFunctionDefArguments of the function.

        Return a list of all the arguments passed to the function.
        These objects all have the type BindCFunctionDefArgument so
        shapes and strides are hidden.
        """
        return self._arguments

    @property
    def bind_c_results(self):
        """
        Get the BindCFunctionDefResults of the function.

        Return a list of all the results returned by the function.
        These objects all have the type BindCFunctionDefResult so
        shapes and strides are hidden.
        """
        return self._results

    @property
    def results(self):
        result_packs = [[r.var, *r.sizes] for r in self._results]
        return [r for rp in result_packs for r in rp]

    @property
    def arguments(self):
        return [ai for a in self._arguments for ai in a.get_all_function_def_arguments()]

# =======================================================================================


class BindCFunctionDefArgument(FunctionDefArgument):
    __slots__ = ('_sizes', '_strides', '_original_arg_var')
    _attribute_nodes = FunctionDefArgument._attribute_nodes + ('_sizes', '_strides', '_original_arg_var')

    def __init__(self, var, scope, original_arg_var, **kwargs):
        name = var.name
        rank = original_arg_var.rank
        sizes   = [Variable(dtype=NativeInteger(),
                            name=scope.get_new_name(f'{name}_shape_{i+1}'))
                   for i in range(rank)]
        strides = [Variable(dtype=NativeInteger(),
                            name=scope.get_new_name(f'{name}_stride_{i+1}'))
                   for i in range(rank)]
        self._sizes = sizes
        self._strides = strides
        self._original_arg_var = original_arg_var
        super().__init__(var, **kwargs)

    @property
    def original_function_argument_variable(self):
        return self._original_arg_var

    @property
    def sizes(self):
        return self._sizes

    @property
    def strides(self):
        return self._strides

    def get_all_function_def_arguments(self):
        args = [self]
        args += [FunctionDefArgument(size) for size in self.sizes]
        args += [FunctionDefArgument(stride) for stride in self.strides]
        return args

    def __repr__(self):
        if self.has_default:
            argument = str(self.name)
            value = str(self.value)
            return 'BindCFunctionDefArgument({0}={1})'.format(argument, value)
        else:
            return 'BindCFunctionDefArgument({})'.format(repr(self.name))

    @property
    def inout(self):
        """
        Indicates whether the argument may be modified by the function.

        True if the argument may be modified in the function. False if
        the argument remains constant in the function.
        """
        return [super().inout, False, False]

# =======================================================================================


class BindCFunctionDefResult(Basic):
    __slots__ = ('_var', '_sizes')
    _attribute_nodes = ('_var', '_sizes')

    def __init__(self, var, sizes = (), **kwargs):
        self._var = var
        self._sizes = sizes
        assert len(sizes) == var.rank
        super().__init__()

    @property
    def var(self):
        return self._var

    @property
    def sizes(self):
        return self._sizes

# =======================================================================================

class BindCModule(Module):
    __slots__ = ('_orig_mod',)
    _attribute_nodes = ('_orig_mod',)

    def __init__(self, *args, original_module, **kwargs):
        self._orig_mod = original_module
        super().__init__(*args, **kwargs)

    @property
    def original_module(self):
        """ The module which was wrapped
        """
        return self._orig_mod

# =======================================================================================
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

# =======================================================================================


def as_static_function(func, *, mod_scope, name=None):
    """
    Translate a FunctionDef to a BindCFunctionDef.

    Translate a FunctionDef to a BindCFunctionDef by altering the
    arguments to allow the function to be called from C.
    E.g. the size of each dimension of an array is provided.

    Parameters
    ----------
    func : FunctionDef
        The function to be translated.
    mod_scope : Scope
        The scope of the module which contains the function.
    name : str
        The new name of the function.

    Returns
    -------
    BindCFunctionDef
        The function which can be called from C.
    """

    assert (isinstance(func, FunctionDef))

    args    = list(func.arguments)
    results = [r.var for r in func.results]
    body    = func.body.body
    functions = func.functions
    _results = []
    interfaces = func.interfaces

    scope = mod_scope.new_child_scope(func.name)

    # Convert array results to inout arguments
    for r in results:
        if r.rank > 0 and r not in args:
            # wrap the array that is returned from the original function
            array_body, array_vars = wrap_array(r, scope, False)
            scope.insert_variable(array_vars[-1])
            scope.insert_variable(r)
            body = body + tuple(array_body)
            array_vars.pop(-1)
            _results += array_vars
        elif r.rank == 0:
            _results += [r]

    body = CodeBlock(body)

    if name is None:
        name = 'bind_c_{}'.format(func.name).lower()

    # ...
    results_names = [i.name for i in _results]
    _args = []

    for i_a, a in enumerate(args):
        a = a.var
        if not isinstance(a, (Variable, FunctionAddress)):
            raise TypeError(
                'Expecting a Variable or FunctionAddress type for {}'.format(a))
        if not isinstance(a, FunctionAddress) and a.rank > 0:
            # ...
            additional_args = []
            for i in range(a.rank):
                n_name = 'n{i}_{name}'.format(name=a.name, i=i)
                n_arg = Variable('int', n_name)

                additional_args += [n_arg]

            shape_new = tuple(additional_args)
            # ...

            _args += [FunctionDefArgument(a) for a in additional_args]

            a_new = Variable( a.dtype, a.name,
                              memory_handling = a.memory_handling,
                              is_argument = True,
                              is_optional = a.is_optional,
                              shape       = shape_new,
                              rank        = a.rank,
                              order       = a.order,
                              precision   = a.precision)

            _args.append(FunctionDefArgument(a_new))

        else:
            _args.append(FunctionDefArgument(a))

    args = _args
    results = [FunctionDefResult(r) for r in _results]
    # ...
    return BindCFunctionDef( name, args, results, body,
                        is_static = True,
                        functions = functions,
                        interfaces = interfaces,
                        imports = func.imports,
                        original_function = func,
                        doc_string = func.doc_string,
                        scope = scope
                        )

# =======================================================================================

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
    scope = Scope(used_symbols=original_module.scope.local_used_symbols.copy())
    bind_c_funcs = [as_static_function_call(
        f, original_module, scope, imports=imports) for f in funcs]
    bind_c_arrays = [wrap_module_array_var(
        v, scope, original_module) for v in variables if v.rank > 0]
    if isinstance(original_module.name, AsName):
        name = scope.get_new_name(
            'bind_c_{}'.format(original_module.name.target))
    else:
        name = scope.get_new_name('bind_c_{}'.format(original_module.name))
    return Module(name, (), bind_c_funcs+bind_c_arrays, imports=imports, scope=scope)

#=======================================================================================
def as_static_function_call(func, mod, mod_scope, name=None, imports = None):
    """
    Create a BindCFunctionDef which calls the provided FunctionDef.

    Translate a FunctionDef to a BindCFunctionDef which calls the
    original function. A BindCFunctionDef is a FunctionDef where the
    arguments are altered to allow the function to be called from c.
    E.g. the size of each dimension of an array is provided.

    Parameters
    ----------
    func : FunctionDef
        The function to be translated.

    mod : Module
        The module which contains the function.

    mod_scope : Scope
        The scope describing the module.

    name : str
        The new name of the function.

    imports : list
        An optional parameter into which any required imports
        can be collected.

    Returns
    -------
    BindCFunctionDef
        The function which can be called from C.
    """

    assert isinstance(func, FunctionDef)
    assert isinstance(mod, Module)
    mod_name = mod.scope.get_python_name(mod.name)

    # from module import func
    if imports is None:
        local_imports = [Import(target=AsName(
            func, func.name), source=mod_name, mod=mod)]
    else:
        imports.append(Import(target=AsName(
            func, func.name), source=mod_name, mod=mod))
        local_imports = ()

    # function arguments
    args = sanitize_arguments(func.arguments)
    # function body
    call    = FunctionCall(func, args)
    results = [r.var for r in func.results]
    results = results[0] if len(results) == 1 else results
    stmt = call if len(func.results) == 0 else Assign(results, call)
    body = [stmt]

    # new function declaration
    new_func = FunctionDef(func.name, func.arguments, func.results, body,
                           functions=func.functions,
                           interfaces=func.interfaces,
                           imports=local_imports,
                           doc_string=func.doc_string
                           )

    # make it compatible with c
    static_func = as_static_function(new_func, name=name, mod_scope=mod_scope)

    return static_func

# =======================================================================================


class BindCPointer(DataType):
    """ Datatype representing a c pointer in fortran
    """
    __slots__ = ()
    _name = 'bindcpointer'

# =======================================================================================


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

# =======================================================================================

class C_F_Pointer(Basic):
    __slots__ = ('_c_expr', '_f_expr', '_sizes')
    _attribute_nodes = ('_c_expr', '_f_expr', '_sizes')

    def __init__(self, c_expr, f_expr, sizes):
        self._c_expr = c_expr
        self._f_expr = f_expr
        self._sizes = sizes
        super().__init__()

    @property
    def c_pointer(self):
        return self._c_expr

    @property
    def f_array(self):
        return self._f_expr

    @property
    def sizes(self):
        return self._sizes

# =======================================================================================


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
    bind_var = Variable(dtype=BindCPointer(),
                        name=scope.get_new_name('bound_'+var.name))
    sizes = [Variable(dtype=NativeInteger(), name=scope.get_new_name())
             for _ in range(var.rank)]
    assigns = [Assign(sizes[i], var.shape[i]) for i in range(var.rank)]
    variables = [bind_var, *sizes]
    if not persistent:
        ptr_var = var.clone(scope.get_new_name(var.name+'_ptr'),
                            memory_handling='alias')
        alloc = Allocate(ptr_var, shape=var.shape,
                         order=var.order, status='unallocated')
        copy = Assign(ptr_var, var)
        c_loc = CLocFunc(ptr_var, bind_var)
        variables.append(ptr_var)
        body = [*assigns, alloc, copy, c_loc]
    else:
        c_loc = CLocFunc(var, bind_var)
        body = [*assigns, c_loc]
    return body, variables

# =======================================================================================


def wrap_module_array_var(var, scope, mod):
    """
    Get a function which allows a module variable to be accessed.

    Create a function which exposes a module array variable to C.
    This allows the module variable to be accessed from the Python
    code.

    Parameters
    ----------
    var : Variable
            The array to be exposed.
    scope : Scope
            The current scope (used to find valid names for variables).
    mod : Module
            The module where the variable is defined.

    Returns
    -------
    FunctionDef
        A function which wraps an array and can be called from C.
    """
    func_name = 'bind_c_'+var.name.lower()
    func_scope = scope.new_child_scope(func_name)
    body, necessary_vars = wrap_array(var, func_scope, True)
    func_scope.insert_variable(necessary_vars[0])
    arg_vars = necessary_vars
    result_vars = [FunctionDefResult(v) for v in necessary_vars]
    import_mod = Import(mod.name, AsName(var,var.name), mod=mod)
    func = BindCFunctionDef(name = func_name,
                  body      = body,
                  arguments = [],
                  results   = result_vars,
                  imports   = [import_mod],
                  scope = func_scope,
                  original_function = None)
    return func
