# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Module describing the code-wrapping class : CToPythonWrapper
which creates an interface exposing C code to Python.
"""
from pyccel.ast.bind_c        import BindCFunctionDef, BindCPointer, BindCFunctionDefArgument
from pyccel.ast.bind_c        import BindCModule
from pyccel.ast.core          import Interface, If, IfSection, Return, FunctionCall
from pyccel.ast.core          import FunctionDef, FunctionDefArgument, FunctionDefResult
from pyccel.ast.core          import Assign, AliasAssign, Deallocate, Allocate
from pyccel.ast.core          import Import, Module, AugAssign, CommentBlock
from pyccel.ast.cwrapper      import PyModule, PyccelPyObject, PyArgKeywords
from pyccel.ast.cwrapper      import PyArg_ParseTupleNode, Py_None
from pyccel.ast.cwrapper      import py_to_c_registry, check_type_registry, PyBuildValueNode
from pyccel.ast.cwrapper      import PyErr_SetString, PyTypeError
from pyccel.ast.cwrapper      import C_to_Python, PyFunctionDef, PyInterface
from pyccel.ast.c_concepts    import ObjectAddress
from pyccel.ast.datatypes     import NativeFloat, NativeComplex
from pyccel.ast.datatypes     import NativeVoid
from pyccel.ast.internals     import get_final_precision
from pyccel.ast.literals      import Nil, LiteralTrue, LiteralString, LiteralInteger
from pyccel.ast.numpy_wrapper import pyarray_to_ndarray, array_type_check
from pyccel.ast.numpy_wrapper import array_get_data, array_get_dim
from pyccel.ast.numpy_wrapper import array_get_c_step, array_get_f_step
from pyccel.ast.numpy_wrapper import numpy_dtype_registry, numpy_flag_f_contig, numpy_flag_c_contig
from pyccel.ast.numpy_wrapper import pyarray_check, is_numpy_array, no_order_check
from pyccel.ast.operators     import PyccelNot, PyccelIsNot, PyccelUnarySub, PyccelEq
from pyccel.ast.variable      import Variable, DottedVariable
from pyccel.parser.scope      import Scope
from pyccel.errors.errors     import Errors
from pyccel.errors.messages   import PYCCEL_RESTRICTION_TODO
from .wrapper                 import Wrapper

errors = Errors()

cwrapper_ndarray_import = Import('cwrapper_ndarrays', Module('cwrapper_ndarrays', (), ()))

class CToPythonWrapper(Wrapper):
    """
    Class for creating a wrapper exposing C code to Python.

    A class which provides all necessary functions for wrapping different AST
    objects such that the resulting AST is Python-compatible.
    """
    def __init__(self):
        # A map used to find the Python-compatible Variable equivalent to an object in the AST
        self._python_object_map = {}
        # Indicate if arrays were wrapped.
        self._wrapping_arrays = False
        super().__init__()

    def get_new_PyObject(self, name, is_temp = False):
        """
        Create new `PyccelPyObject` `Variable` with the desired name.

        Create a new `Variable` with the datatype `PyccelPyObject` and the desired name.
        A `PyccelPyObject` datatype means that this variable can be accessed and
        manipulated from Python.

        Parameters
        ----------
        name : str
            The desired name.

        is_temp : bool, default=False
            Indicates if the Variable is temporary. A temporary variable may be ignored
            by the printer.

        Returns
        -------
        Variable
            The new variable.
        """
        var = Variable(dtype=PyccelPyObject(),
                        name=self.scope.get_new_name(name),
                        memory_handling='alias',
                        is_temp=is_temp)
        self.scope.insert_variable(var)
        return var

    def _get_python_argument_variables(self, args):
        """
        Get a new set of `PyccelPyObject` `Variable`s representing each of the arguments.

        Create a new `PyccelPyObject` variable for each argument returned in Python.
        The results are saved to the `self._python_object_map` dictionary so they can be
        discovered later.

        Parameters
        ----------
        args : iterable of FunctionDefArguments
            The arguments of the function.

        Returns
        -------
        list of Variable
            Variables which will hold the arguments in Python.
        """
        collect_args = [self.get_new_PyObject(a.var.name+'_obj') for a in args]
        self._python_object_map.update(dict(zip(args, collect_args)))
        return collect_args

    def _unpack_python_args(self, args):
        """
        Unpack the arguments received from Python into the expected Python variables.

        Create the wrapper arguments of the current `FunctionDef` (`self`, `args`, `kwargs`).
        Get a new set of `PyccelPyObject` `Variable`s representing each of the expected
        arguments. Add the code which unpacks the `args` and `kwargs` into individual
        `PyccelPyObject`s for each of the expected arguments.

        Parameters
        ----------
        args : iterable of FunctionDefArguments
            The expected arguments of the function.

        Returns
        -------
        func_args : list of Variable
            The arguments of the FunctionDef.

        body : list of pyccel.ast.basic.Basic
            The code which unpacks the arguments.

        Examples
        --------
        >>> arg = Variable('int', 'x')
        >>> func_args = (FunctionDefArgument(arg),)
        >>> wrapper_args, body = self._unpack_python_args(func_args)
        >>> wrapper_args
        [Variable('self', dtype=PyccelPyObject()), Variable('args', dtype=PyccelPyObject()), Variable('kwargs', dtype=PyccelPyObject())]
        >>> body
        [<pyccel.ast.cwrapper.PyArgKeywords object at 0x7f99ec128cc0>, <pyccel.ast.core.If object at 0x7f99ed3a5b20>]
        >>> CWrapperCodePrinter('wrapper_file.c').doprint(expr)
        static char *kwlist[] = {
            "x",
            NULL
        };
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &x_obj))
        {
            return NULL;
        }
        """
        # Create necessary variables
        func_args = [self.get_new_PyObject(n) for n in ("self", "args", "kwargs")]
        arg_vars  = self._get_python_argument_variables(args)
        keyword_list_name = self.scope.get_new_name('kwlist')

        # Create the list of argument names
        arg_names = [getattr(a, 'original_function_argument_variable', a.var).name for a in args]
        keyword_list = PyArgKeywords(keyword_list_name, arg_names)

        # Initialise optionals
        body = [AliasAssign(py_arg, Py_None) for func_def_arg, py_arg in zip(args, arg_vars) if func_def_arg.has_default]

        # Parse arguments
        parse_node = PyArg_ParseTupleNode(*func_args[1:], args, arg_vars, keyword_list)

        body.append(keyword_list)
        body.append(If(IfSection(PyccelNot(parse_node), [Return([Nil()])])))

        return func_args, body

    def _get_python_result_variables(self, results):
        """
        Get a new set of `PyccelPyObject` `Variable`s representing each of the results.

        Create a new `PyccelPyObject` variable for each result returned in Python.
        The results are saved to the `self._python_object_map` dictionary so they can be
        discovered later.

        Parameters
        ----------
        results : iterable of FunctionDefResults
            The results of the function.

        Returns
        -------
        list of Variable
            Variables which will hold the results in Python.
        """
        collect_results = [self.get_new_PyObject(r.var.name+'_obj') for r in results]
        self._python_object_map.update(dict(zip(results, collect_results)))
        return collect_results

    def _get_check_function(self, py_obj, arg, raise_error):
        """
        Get the function which checks if an argument has the expected type.

        Using the c-compatible description of a function argument, determine whether the Python
        object (with datatype `PyccelPyObject`) holds data which is compatible with the expected
        type. The check is returned along with any errors that may be raised depending upon the
        result and the value of `raise_error`.

        Parameters
        ----------
        py_obj : Variable
            The variable with datatype `PyccelPyObject` where the arguments is stored in Python.

        arg : Variable
            The c-compatible variable which holds all the details about the expected type.

        raise_error : bool
            True if an error should be raised in case of an unexpected type, False otherwise.

        Returns
        -------
        func_call : FunctionCall
            The function call which checks if the argument has the expected type.

        error_code : tuple of pyccel.ast.basic.Basic
            The code which raises any necessary errors.
        """
        rank = arg.rank
        error_code = ()
        if rank == 0:
            dtype = arg.dtype
            prec  = arg.precision
            try :
                cast_function = check_type_registry[(dtype, prec)]
            except KeyError:
                errors.report(f"Can't check the type of {dtype}[kind = {prec}]\n"+PYCCEL_RESTRICTION_TODO,
                        symbol=arg, severity='fatal')
            func = FunctionDef(name = cast_function,
                               body      = [],
                               arguments = [FunctionDefArgument(Variable(dtype=PyccelPyObject(), name = 'o', memory_handling='alias'))],
                               results   = [FunctionDefResult(Variable(dtype=dtype, name = 'v', precision = prec))])

            if raise_error:
                message = LiteralString(f"Expected an argument of type {dtype} for argument {arg.name}")
                python_error = FunctionCall(PyErr_SetString, [PyTypeError, message])
                error_code = (python_error,)

            func_call = FunctionCall(func, [py_obj])
        else:
            dtype = str(arg.dtype)
            prec  = get_final_precision(arg)
            try :
                 type_ref = numpy_dtype_registry[(dtype, prec)]
            except KeyError:
                errors.report(f"Can't check the type of an array of {dtype}[kind = {prec}]\n"+PYCCEL_RESTRICTION_TODO,
                        symbol=arg, severity='fatal')

            # order flag
            if rank == 1:
                flag     = no_order_check
            elif arg.order == 'F':
                flag = numpy_flag_f_contig
            else:
                flag = numpy_flag_c_contig

            check_func = pyarray_check if raise_error else is_numpy_array
            # No error code required as the error is raised inside pyarray_check

            func_call = FunctionCall(check_func, [py_obj, type_ref, LiteralInteger(rank), flag])

        return func_call, error_code

    def _get_type_check_function(self, name, args, funcs):
        """
        Determine the flags which allow correct function to be identified from the interface.

        Each function must be identifiable by a different integer value. This value is known
        as a flag. Different parts of the flag to indicate the types of different arguments.
        Take for example the following function:
        ```python
        @types('int', 'int')
        @types('float', 'float')
        def f(a, b):
            pass
        ```
        The values 0 (int) and 1 (float) would indicate the type of the argument a. In order
        to preserve this information the values which indicate the type of the argument b
        must only change the part of the flag which does not contain this information. In other
        words `flag % n_types_a = flag_a`. Therefore the values 0 (int) and 2(float) indicate
        the type of the argument b.
        We then finally have the following four options:
          1. 0 = 0 + 0 => (int,int)
          2. 1 = 1 + 0 => (float,int)
          3. 2 = 0 + 2 => (int, float)
          4. 3 = 1 + 2 => (float, float)

        of which only the first and last flags indicate acceptable arguments.

        The function returns a dictionary whose keys are the functions and whose values are
        a list of the flags which would indicate the correct types.
        In the above example we would return `{func_0 : [0,0], func_1 : [1,2]}`.
        It also returns a FunctionDef which determines the index of the chosen function.

        Parameters
        ----------
        funcs : list of FunctionDefs
            The functions in the Interface.

        Returns
        -------
        func : FunctionDef
            The function which determines the key identifying the relevant function.
        argument_type_flags : dict
            A dictionary whose keys are the functions and whose values are the integer keys
            which indicate that the function should be chosen.
        """
        func_scope = self.scope.new_child_scope(name)
        self.scope = func_scope
        orig_funcs = [getattr(func, 'original_function', func) for func in funcs]
        type_indicator = Variable('int', self.scope.get_new_name('type_indicator'))

        # Initialise the argument_type_flags
        argument_type_flags = {func : 0 for func in funcs}

        # Initialise type_indicator
        body = [Assign(type_indicator, LiteralInteger(0))]

        step = 1
        for i, py_arg in enumerate(args):
            # Get the relevant typed arguments from the original functions
            interface_args = [getattr(func.arguments[i], 'original_function_argument_variable', func.arguments[i].var) for func in orig_funcs]
            # Get the type key
            interface_types = [(a.dtype, a.precision, a.rank, a.order) for a in interface_args]
            # Get a dictionary mapping each unique type key to an example argument
            type_to_example_arg = dict(zip(interface_types, interface_args))
            # Get a list of unique keys
            possible_types = list(type_to_example_arg.keys())

            n_possible_types = len(possible_types)
            if n_possible_types != 1:
                # Update argument_type_flags with the index of the type key
                for func, t in zip(funcs, interface_types):
                    index = possible_types.index(t)*step
                    argument_type_flags[func] += index

                # Create the type checks and incrementation of the type_indicator
                if_blocks = []
                for index, t in enumerate(possible_types):
                    check_func_call, _ = self._get_check_function(py_arg, type_to_example_arg[t], False)
                    if_blocks.append(IfSection(check_func_call, [AugAssign(type_indicator, '+', LiteralInteger(index*step))]))
                body.append(If(*if_blocks, IfSection(LiteralTrue(), 
                            [FunctionCall(PyErr_SetString, [PyTypeError, f"Unexpected type for argument {interface_args[0].name}"]),
                             Return([PyccelUnarySub(LiteralInteger(1))])])))

            # Update the step to ensure unique indices for each argument
            step *= n_possible_types

        body.append(Return([type_indicator]))

        self.exit_scope()

        doc_string = CommentBlock("Assess the types. Raise an error for unexpected types and calculate an integer\n" + 
                        "which indicates which function should be called.")

        # Build the function
        func = FunctionDef(name, [FunctionDefArgument(a) for a in args], [FunctionDefResult(type_indicator)],
                            body, doc_string=doc_string, scope=func_scope)

        return func, argument_type_flags

    #--------------------------------------------------------------------------------------------------------------------------------------------

    def _wrap_Module(self, expr):
        """
        Build a `PyModule` from a `Module`.

        Create a `PyModule` which wraps a C-compatible `Module`.

        Parameters
        ----------
        expr : Module
            The module which can be called from C.

        Returns
        -------
        PyModule
            The module which can be called from Python.
        """
        # Define scope
        scope = expr.scope
        mod_scope = Scope(used_symbols = scope.local_used_symbols.copy(), original_symbols = scope.python_names.copy())
        self.scope = mod_scope
        classes = [self._wrap(i) for i in expr.classes]
        funcs_to_wrap = [f for f in expr.funcs if f not in (expr.init_func, expr.free_func) and not f.is_private]
        funcs = [self._wrap(f) for f in funcs_to_wrap]
        interfaces = [self._wrap(i) for i in expr.interfaces]
        variables = [self._wrap(v) for v in (expr.original_module.variables if isinstance(expr, BindCModule) else expr.variables)]
        self.exit_scope()
        imports = (cwrapper_ndarray_import,) if self._wrapping_arrays else ()
        original_mod = getattr(expr, 'original_module', expr)
        return PyModule(original_mod.name, (), funcs, func_names = (), imports = imports,
                        interfaces = interfaces, scope = mod_scope, external_funcs = funcs_to_wrap)

    def _wrap_Interface(self, expr):
        func_name = self.scope.get_new_name(expr.name+'_wrapper')
        func_scope = self.scope.new_child_scope(func_name)
        self.scope = func_scope
        original_funcs = expr.functions
        example_func = original_funcs[0]

        # Add the variables to the expected symbols in the scope
        for a in getattr(example_func, 'bind_c_arguments', example_func.arguments):
            func_scope.insert_symbol(a.var.name)
        python_args = getattr(example_func, 'bind_c_arguments', example_func.arguments)
        func_args, body = self._unpack_python_args(python_args)

        python_arg_objs = [self._python_object_map[a] for a in python_args]

        type_indicator = Variable('int', self.scope.get_new_name('type_indicator'))
        self.scope.insert_variable(type_indicator)

        self.exit_scope()

        # Determine flags which indicate argument type
        type_check_name = self.scope.get_new_name(expr.name+'_type_check')
        type_check_func, argument_type_flags = self._get_type_check_function(type_check_name, python_arg_objs, original_funcs)

        body.append(Assign(type_indicator, FunctionCall(type_check_func, python_arg_objs)))

        functions = []
        if_sections = []
        for func, index in argument_type_flags.items():
            wrapped_func = self._python_object_map[func]
            if_sections.append(IfSection(PyccelEq(type_indicator, LiteralInteger(index)),
                                [Return([FunctionCall(wrapped_func, python_arg_objs)])]))
            functions.append(wrapped_func)
        if_sections.append(IfSection(LiteralTrue(), 
                    [FunctionCall(PyErr_SetString, [PyTypeError, f"Unexpected type combination"]),
                     Return([Nil()])]))
        body.append(If(*if_sections))

        interface_func = FunctionDef(func_name,
                                     [FunctionDefArgument(a) for a in func_args],
                                     [FunctionDefResult(self.get_new_PyObject("result", is_temp=True))],
                                     body,
                                     scope=func_scope)
        for a in python_args:
            self._python_object_map.pop(a)

        return PyInterface(func_name, interface_func, type_check_func, functions, expr)

    def _wrap_FunctionDef(self, expr):
        """
        Build a `PyFunctionDef` form a `FunctionDef`.

        Create a `PyFunctionDef` which wraps a C-compatible `FunctionDef`.
        The `PyFunctionDef` should take three arguments (`self`, `args`,
        and `kwargs`) and return a `PyccelPyObject`. If the function is
        called from an Interface then the arguments are `PyccelPyObject`s
        describing each of the arguments of the C-compatible function.

        Parameters
        ----------
        expr : FunctionDef
            The function which can be called from C.

        Returns
        -------
        PyFunctionDef
            The function which can be called from Python.
        """
        original_func = getattr(expr, 'original_function', expr)
        func_name = self.scope.get_new_name(original_func.name+'_wrapper')
        func_scope = self.scope.new_child_scope(func_name)
        self.scope = func_scope

        is_bind_c_function_def = isinstance(expr, BindCFunctionDef)

        # Add the variables to the expected symbols in the scope
        for a in expr.arguments:
            func_scope.insert_symbol(a.var.name)
        for a in getattr(expr, 'bind_c_arguments', ()):
            func_scope.insert_symbol(a.original_function_argument_variable.name)
        for r in expr.results:
            func_scope.insert_symbol(r.var.name)

        in_interface = len(expr.get_user_nodes(Interface)) > 0

        # Get variables describing the arguments and results that are seen from Python
        python_args = expr.bind_c_arguments if is_bind_c_function_def else expr.arguments
        python_results = expr.bind_c_results if is_bind_c_function_def else expr.results

        # Get variables describing the arguments and results that must be passed to the function
        original_c_args = expr.arguments
        original_c_results = expr.results

        # Get the arguments of the PyFunctionDef
        if in_interface:
            func_args = [FunctionDefArgument(a) for a in self._get_python_argument_variables(python_args)]
            body = []
        else:
            func_args, body = self._unpack_python_args(python_args)
            func_args = [FunctionDefArgument(a) for a in func_args]

        # Get the results of the PyFunctionDef
        python_result_variables = self._get_python_result_variables(python_results)

        # Get the code required to extract the C-compatible arguments from the Python arguments
        body += [l for a in python_args for l in self._wrap(a)]

        # Get the code required to wrap the C-compatible results into Python objects
        # This function creates variables so it must be called before extracting them from the scope.
        result_wrap = [l for r in python_results for l in self._wrap(r)]

        # Get the names of the arguments which should be used to call the C-compatible function
        func_call_arg_names = []
        for a in original_c_args:
            if isinstance(a, BindCFunctionDefArgument):
                orig_var = a.original_function_argument_variable
                if orig_var.is_optional and not orig_var.is_ndarray:
                    func_call_arg_names.append(orig_var.name)
                    continue
            func_call_arg_names.append(a.var.name)

        # Get the arguments and results which should be used to call the c-compatible function
        func_call_args = [self.scope.find(self.scope.get_expected_name(n), category='variables') for n in func_call_arg_names]
        c_results = [self.scope.find(self.scope.get_expected_name(r.var.name), category='variables') for r in original_c_results]
        c_results = [ObjectAddress(r) if r.dtype is BindCPointer() else r for r in c_results]

        # Call the C-compatible function
        n_c_results = len(c_results)
        if n_c_results == 0:
            body.append(FunctionCall(expr, func_call_args))
        elif n_c_results == 1:
            body.append(Assign(c_results[0], FunctionCall(expr, func_call_args)))
        else:
            body.append(Assign(c_results, FunctionCall(expr, func_call_args)))

        # Deallocate the C equivalent of any array arguments
        # The C equivalent is the same variable that is passed to the function unless the target language is Fortran.
        # In this case the function arguments are the data pointer and the shapes and strides, but the C equivalent
        # is an ndarray.
        for a in original_c_args:
            orig_var = getattr(a, 'original_function_argument_variable', a.var)
            v = self.scope.find(self.scope.get_expected_name(orig_var.name), category='variables')
            if v.is_ndarray:
                if v.is_optional:
                    body.append(If( IfSection(PyccelIsNot(v, Nil()), [Deallocate(v)]) ))
                else:
                    body.append(Deallocate(v))
        body.extend(result_wrap)

        # Pack the Python compatible results of the function into one argument.
        n_py_results = len(python_result_variables)
        if n_py_results == 0:
            res = Py_None
            func_results = [FunctionDefResult(self.get_new_PyObject("result", is_temp=True))]
        elif n_py_results == 1:
            res = python_result_variables[0]
            func_results = [FunctionDefResult(res)]
        else:
            res = self.get_new_PyObject("result")
            body.append(AliasAssign(res, PyBuildValueNode([ObjectAddress(r) for r in python_result_variables])))
            func_results = [FunctionDefResult(res)]
        body.append(Return([res]))

        self.exit_scope()
        for a in python_args:
            self._python_object_map.pop(a)
        for r in python_results:
            self._python_object_map.pop(r)

        function = PyFunctionDef(func_name, func_args, func_results, body, scope=func_scope, original_function = original_func)

        self.scope.functions[func_name] = function
        self._python_object_map[expr] = function

        return function

    def _wrap_FunctionDefArgument(self, expr):

        collect_arg = self._python_object_map[expr]
        in_interface = len(expr.get_user_nodes(Interface)) > 0

        orig_var = getattr(expr, 'original_function_argument_variable', expr.var)
        if orig_var.is_ndarray:
            arg_var = orig_var.clone(self.scope.get_expected_name(orig_var.name), is_argument = False, memory_handling='alias')
            self._wrapping_arrays = True
        else:
            arg_var = orig_var.clone(self.scope.get_expected_name(orig_var.name), is_argument = False)

        self.scope.insert_variable(arg_var)

        body = []

        # Initialise to any default value
        if expr.has_default:
            default_val = expr.value
            if isinstance(default_val, Nil):
                body.append(AliasAssign(arg_var, default_val))
            else:
                body.append(Assign(arg_var, default_val))

        # Collect the function which casts from a Python object to a C object
        if arg_var.rank == 0:
            dtype = arg_var.dtype
            prec  = get_final_precision(arg_var)
            try :
                cast_function = py_to_c_registry[(dtype, prec)]
            except KeyError:
                errors.report(PYCCEL_RESTRICTION_TODO, symbol=dtype,severity='fatal')
            cast_func = FunctionDef(name = cast_function,
                               body      = [],
                               arguments = [FunctionDefArgument(Variable(dtype=PyccelPyObject(), name = 'o', memory_handling='alias'))],
                               results   = [FunctionDefResult(Variable(dtype=dtype, name = 'v', precision = prec))])
        else:
            cast_func = pyarray_to_ndarray

        cast = Assign(arg_var, FunctionCall(cast_func, [collect_arg]))

        # Create any necessary type checks and errors
        if expr.has_default:
            check_func, err = self._get_check_function(collect_arg, arg_var, False)
            body.append(If( IfSection(check_func, [cast]),
                        IfSection(PyccelIsNot(collect_arg, Py_None), [*err, Return([Nil()])])
                        ))
        elif not in_interface:
            check_func, err = self._get_check_function(collect_arg, arg_var, True)
            body.append(If( IfSection(check_func, [cast]),
                        IfSection(LiteralTrue(), [*err, Return([Nil()])])
                        ))
        else:
            body.append(cast)

        return body

    def _wrap_BindCFunctionDefArgument(self, expr):
        body = self._wrap_FunctionDefArgument(expr)

        orig_var = expr.original_function_argument_variable

        if orig_var.rank:
            arg_var = expr.var.clone(self.scope.get_expected_name(expr.var.name), is_argument = False)
            shape_vars = [s.clone(self.scope.get_expected_name(s.name), is_argument = False) for s in expr.shape]
            stride_vars = [s.clone(self.scope.get_expected_name(s.name), is_argument = False) for s in expr.strides]
            self.scope.insert_variable(arg_var, expr.var.name)
            for v,s in zip(shape_vars, expr.shape):
                self.scope.insert_variable(v,s.name)
            for v,s in zip(stride_vars, expr.strides):
                self.scope.insert_variable(v,s.name)

            c_arg = self.scope.find(self.scope.get_expected_name(orig_var.name), category='variables')

            body.append(AliasAssign(arg_var, FunctionCall(array_get_data, [c_arg])))
            body.extend(Assign(s, FunctionCall(array_get_dim, [c_arg, i])) for i,s in enumerate(shape_vars))
            if orig_var.order == 'C':
                body.extend(Assign(s, FunctionCall(array_get_c_step, [c_arg, i])) for i,s in enumerate(stride_vars))
            else:
                body.extend(Assign(s, FunctionCall(array_get_f_step, [c_arg, i])) for i,s in enumerate(stride_vars))

        return body

    def _wrap_FunctionDefResult(self, expr):

        orig_var = expr.var

        if orig_var.is_ndarray:
            c_res = orig_var.clone(self.scope.get_expected_name(orig_var.name), is_argument = False, memory_handling='alias')
            self._wrapping_arrays = True
        else:
            c_res = orig_var.clone(self.scope.get_expected_name(orig_var.name), is_argument = False)
        self.scope.insert_variable(c_res)

        python_res = self._python_object_map[expr]
        body = [AliasAssign(python_res, FunctionCall(C_to_Python(c_res), [ObjectAddress(c_res)]))]

        if orig_var.rank:
            body.append(Deallocate(c_res))

        return body

    def _wrap_BindCFunctionDefResult(self, expr):

        orig_var = expr.original_function_result_variable

        body = []

        if orig_var.rank:
            # C-compatible result variable
            c_res = orig_var.clone(self.scope.get_new_name(orig_var.name), is_argument = False, memory_handling='alias')
            self._wrapping_arrays = True
            # Result of calling the bind-c function
            arg_var = expr.var.clone(self.scope.get_expected_name(expr.var.name), is_argument = False, memory_handling='alias')
            shape_vars = [s.clone(self.scope.get_expected_name(s.name), is_argument = False) for s in expr.shape]
            # Save so we can find by iterating over func.results
            self.scope.insert_variable(arg_var, expr.var.name)
            [self.scope.insert_variable(v,s.name) for v,s in zip(shape_vars, expr.shape)]
            # Save so we can find by iterating over func.bind_c_results
            self.scope.insert_variable(c_res, orig_var.name)

            body.append(Allocate(c_res, shape = shape_vars, order = orig_var.order, status='unallocated'))
            body.append(AliasAssign(DottedVariable(NativeVoid(), 'raw_data', memory_handling = 'alias', lhs=c_res), arg_var))
        else:
            c_res = orig_var.clone(self.scope.get_expected_name(orig_var.name), is_argument = False)
            self.scope.insert_variable(c_res)

        python_res = self._python_object_map[expr]
        body.append(AliasAssign(python_res, FunctionCall(C_to_Python(c_res), [ObjectAddress(c_res)])))

        if orig_var.rank:
            body.append(Deallocate(c_res))

        return body
