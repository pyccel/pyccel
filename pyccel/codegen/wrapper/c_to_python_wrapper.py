# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Module describing the code-wrapping class : CToPythonWrapper
which creates an interface exposing C code to Python.
"""
from pyccel.ast.bind_c        import BindCFunctionDef, BindCPointer
from pyccel.ast.core          import Interface, If, IfSection, Return, FunctionCall
from pyccel.ast.core          import FunctionDef, FunctionDefArgument, FunctionDefResult
from pyccel.ast.core          import Assign, AliasAssign, Deallocate, Allocate
from pyccel.ast.core          import Import, Module
from pyccel.ast.cwrapper      import PyModule, PyccelPyObject, PyArgKeywords
from pyccel.ast.cwrapper      import PyArg_ParseTupleNode, Py_None
from pyccel.ast.cwrapper      import py_to_c_registry, check_type_registry, PyBuildValueNode
from pyccel.ast.cwrapper      import PyErr_SetString, PyTypeError
from pyccel.ast.cwrapper      import C_to_Python, PyFunctionDef
from pyccel.ast.c_concepts    import ObjectAddress
from pyccel.ast.datatypes     import NativeFloat, NativeComplex
from pyccel.ast.datatypes     import NativeVoid
from pyccel.ast.internals     import get_final_precision
from pyccel.ast.literals      import Nil, LiteralTrue, LiteralString
from pyccel.ast.numpy_wrapper import pyarray_to_ndarray, array_type_check
from pyccel.ast.numpy_wrapper import array_get_data, array_get_dim
from pyccel.ast.numpy_wrapper import array_get_c_step, array_get_f_step
from pyccel.ast.operators     import PyccelNot, PyccelIsNot
from pyccel.ast.variable      import Variable, DottedVariable
from pyccel.parser.scope      import Scope
from pyccel.errors.errors     import Errors
from pyccel.errors.messages   import PYCCEL_RESTRICTION_TODO
from .wrapper                 import Wrapper

errors = Errors()

cwrapper_ndarray_import = Import('cwrapper_ndarrays', Module('cwrapper_ndarrays', (), ()))

class CToPythonWrapper(Wrapper):
    def __init__(self):
        self._python_object_map = None
        self._wrapping_arrays = False
        super().__init__()

    def get_new_PyObject(self, name, is_temp = False):
        """
        Create new PyccelPyObject Variable with the desired name.

        Parameters
        ----------
        name : String
            The desired name.

        is_temp : bool
            Indicates if the Variable is temporary.

        Returns
        -------
        Variable
        """
        var = Variable(dtype=PyccelPyObject(),
                        name=self.scope.get_new_name(name),
                        memory_handling='alias',
                        is_temp=is_temp)
        self.scope.insert_variable(var)
        return var

    def _get_python_args(self, args):
        collect_args = [self.get_new_PyObject(a.var.name+'_obj') for a in args]
        self._python_object_map = dict(zip(args, collect_args))

    def _unpack_python_args(self, args):
        func_args = [self.get_new_PyObject(n) for n in ("self", "args", "kwargs")]
        arg_names = [a.var.name for a in args]
        self._get_python_args(args)
        keyword_list_name = self.scope.get_new_name('kwlist')
        keyword_list      = PyArgKeywords(keyword_list_name, arg_names)

        # Parse arguments
        parse_node = PyArg_ParseTupleNode(*func_args[1:], args, list(self._python_object_map.values()), keyword_list)

        body = [keyword_list, If(IfSection(PyccelNot(parse_node), [Return([Nil()])]))]

        return func_args, body

    def _get_check_function(self, py_obj, arg, raise_error):
        if arg.rank == 0:
            dtype = arg.dtype
            prec  = arg.precision
            try :
                cast_function = check_type_registry[(dtype, prec)]
            except KeyError:
                errors.report(PYCCEL_RESTRICTION_TODO, symbol=dtype, severity='fatal')
            func = FunctionDef(name = cast_function,
                               body      = [],
                               arguments = [FunctionDefArgument(Variable(dtype=PyccelPyObject(), name = 'o', memory_handling='alias'))],
                               results   = [FunctionDefResult(Variable(dtype=dtype, name = 'v', precision = prec))])
            message = LiteralString(f"Expected an argument of type {dtype} for argument {arg.name}")
            return FunctionCall(func, [py_obj]), (FunctionCall(PyErr_SetString, [PyTypeError, message]),)
        else:
            return array_type_check(py_obj, arg, raise_error), ()

    def _wrap_Module(self, expr):
        # Define scope
        scope = expr.scope
        mod_scope = Scope(used_symbols = scope.local_used_symbols.copy(), original_symbols = scope.python_names.copy())
        self.scope = mod_scope
        funcs_to_wrap = [f for f in expr.funcs if f not in (expr.init_func, expr.free_func) and not f.is_private]
        funcs = [self._wrap(f) for f in funcs_to_wrap]
        self.exit_scope()
        imports = (cwrapper_ndarray_import,) if self._wrapping_arrays else ()
        original_mod = getattr(expr, 'original_module', expr)
        return PyModule(original_mod.name, (), funcs, func_names = (), imports = imports,
                        scope = mod_scope, external_funcs = funcs_to_wrap)

    def _wrap_FunctionDef(self, expr):
        original_func = getattr(expr, 'original_function', expr)
        func_name = self.scope.get_new_name(original_func.name+'_wrapper')
        func_scope = self.scope.new_child_scope(func_name)
        self.scope = func_scope

        is_bind_c_function_def = isinstance(expr, BindCFunctionDef)

        for a in expr.arguments:
            func_scope.insert_symbol(a.var.name)
        for a in getattr(expr, 'bind_c_arguments', ()):
            func_scope.insert_symbol(a.original_function_argument_variable.name)
        for r in expr.results:
            func_scope.insert_symbol(r.var.name)

        in_interface = len(expr.get_user_nodes(Interface)) > 0

        python_args = expr.bind_c_arguments if is_bind_c_function_def else expr.arguments
        python_results = expr.bind_c_results if is_bind_c_function_def else expr.results

        c_args = expr.arguments
        c_results = expr.results

        if in_interface:
            self._get_python_args(python_args)
            func_args = [FunctionDefArgument(a) for a in self._python_object_map.values()]
            body = []
        else:
            func_args, body = self._unpack_python_args(python_args)
            func_args = [FunctionDefArgument(a) for a in func_args]

        body += [l for a in python_args for l in self._wrap(a)]

        result_wrap = [line for r in python_results for line in self._wrap(r)]

        c_args = [self.scope.find(self.scope.get_expected_name(a.var.name), category='variables') for a in expr.arguments]
        c_results = [self.scope.find(self.scope.get_expected_name(r.var.name), category='variables') for r in expr.results]
        c_results = [ObjectAddress(r) if r.dtype is BindCPointer() else r for r in c_results]

        python_result_variables = [self._python_object_map[r] for r in python_results]

        n_c_results = len(c_results)

        if n_c_results == 0:
            body.append(FunctionCall(expr, c_args))
        elif n_c_results == 1:
            body.append(Assign(c_results[0], FunctionCall(expr, c_args)))
        else:
            body.append(Assign(c_results, FunctionCall(expr, c_args)))

        body.extend(Deallocate(a) for a in c_args if a.is_ndarray)
        body.extend(result_wrap)

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

        return PyFunctionDef(func_name, func_args, func_results, body, scope=func_scope, original_function = original_func)

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
        if arg_var.is_optional:
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

        if orig_var.rank != 0:
            self._wrapping_arrays = True

        if orig_var.is_ndarray:
            c_res = orig_var.clone(self.scope.get_expected_name(orig_var.name), is_argument = False, memory_handling='alias')
            self._wrapping_arrays = True
        else:
            c_res = orig_var.clone(self.scope.get_expected_name(orig_var.name), is_argument = False)
        self.scope.insert_variable(c_res)

        python_res = self.get_new_PyObject(orig_var.name+'_obj')
        self._python_object_map[expr] = python_res

        body = [AliasAssign(python_res, FunctionCall(C_to_Python(c_res), [ObjectAddress(c_res)]))]

        if orig_var.rank:
            body.append(Deallocate(c_res))

        return body

    def _wrap_BindCFunctionDefResult(self, expr):

        orig_var = expr.original_function_result_variable
        python_res = self.get_new_PyObject(orig_var.name+'_obj')

        body = []

        if orig_var.rank:
            # C-compatible result variable
            c_res = orig_var.clone(self.scope.get_new_name(orig_var.name), is_argument = False)
            self._wrapping_arrays = True
            # Result of calling the bind-c function
            arg_var = expr.var.clone(self.scope.get_expected_name(expr.var.name), is_argument = False, memory_handling='alias')
            shape_vars = [s.clone(self.scope.get_expected_name(s.name), is_argument = False) for s in expr.shape]
            # Save so we can find by iterating over func.results
            self.scope.insert_variable(arg_var, expr.var.name)
            [self.scope.insert_variable(v,s.name) for v,s in zip(shape_vars, expr.shape)]
            # Save so we can find by iterating over func.bind_c_results
            self.scope.insert_variable(c_res, orig_var.name)

            body.append(Allocate(c_res, shape = shape_vars, order = orig_var.order,
                status='unallocated'))
            body.append(AliasAssign(DottedVariable(NativeVoid(), 'raw_data', memory_handling = 'alias',
                lhs=c_res), arg_var))
        else:
            c_res = orig_var.clone(self.scope.get_expected_name(orig_var.name), is_argument = False)
            self.scope.insert_variable(c_res)

        self._python_object_map[expr] = python_res

        body.append(AliasAssign(python_res, FunctionCall(C_to_Python(c_res), [ObjectAddress(c_res)])))

        if orig_var.rank:
            body.append(Deallocate(c_res))

        return body
