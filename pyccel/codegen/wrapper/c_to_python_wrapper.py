# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
Module describing the code-wrapping class : CToPythonWrapper
which creates an interface exposing C code to Python.
"""
from pyccel.ast.bind_c   import BindCFunctionDef
from pyccel.ast.core     import Interface, If, IfSection, Return, FunctionCall
from pyccel.ast.core     import FunctionDef, FunctionDefArgument, FunctionDefResult
from pyccel.ast.core     import Assign, AliasAssign, Deallocate, Allocate
from pyccel.ast.core     import Import, Module
from pyccel.ast.cwrapper import PyModule, PyccelPyObject, PyArgKeywords
from pyccel.ast.cwrapper import PyArg_ParseTupleNode, PyccelPyObject, Py_None
from pyccel.ast.cwrapper import py_to_c_registry, check_type_registry, PyBuildValueNode
from pyccel.ast.cwrapper import Python_to_C, PyErr_SetString, PyTypeError
from pyccel.ast.datatypes import NativeInteger, NativeFloat, NativeComplex
from pyccel.ast.datatypes import NativeBool
from pyccel.ast.internals import get_final_precision
from pyccel.ast.literals import Nil, LiteralTrue, LiteralString
from pyccel.ast.operators import PyccelNot, PyccelIsNot
from pyccel.ast.variable import Variable
from pyccel.ast.numpy_wrapper import pyarray_to_ndarray, array_type_check
from pyccel.ast.numpy_wrapper import array_get_data, array_get_dim
from pyccel.ast.numpy_wrapper import array_get_c_step, array_get_f_step
from pyccel.parser.scope import Scope
from pyccel.ast.variable import Variable
from pyccel.errors.errors   import Errors
from pyccel.errors.messages   import PYCCEL_RESTRICTION_TODO
from .wrapper import Wrapper

errors = Errors()

cwrapper_ndarray_import = Import('cwrapper_ndarrays', Module('cwrapper_ndarrays', (), ()))

class CToPythonWrapper(Wrapper):
    def __init__(self):
        self._python_object_map = None
        super().__init__()

    def get_new_PyObject(self, name):
        """
        Create new PyccelPyObject Variable with the desired name.

        Parameters
        ----------
        name : String
            The desired name.

        Returns
        -------
        Variable
        """
        var = Variable(dtype=PyccelPyObject(),
                        name=self.scope.get_new_name(name),
                        memory_handling='alias')
        self.scope.insert_variable(var)
        return var

    def _get_python_args(self, args):
        collect_args = [self.get_new_PyObject(a.var.name+'_obj') for a in args]
        self._python_object_map = {a: c for a,c in zip(args, collect_args)}

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
            prec  = get_final_precision(arg)
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
        return PyModule(expr.name, (), funcs, func_names = (), imports = (cwrapper_ndarray_import,),
                        scope = mod_scope, external_funcs = funcs_to_wrap)

    def _wrap_FunctionDef(self, expr):
        func_name = self.scope.get_new_name(expr.original_function.name+'_wrapper')
        func_scope = self.scope.new_child_scope(func_name)
        self.scope = func_scope

        is_bind_c_function_def = isinstance(expr, BindCFunctionDef)

        for a in expr.arguments:
            func_scope.insert_symbol(a.var.name)
        for a in getattr(expr, 'bind_c_arguments', ()):
            func_scope.insert_symbol(a.original_function_argument_variable.name)
        for r in expr.results:
            func_scope.insert_symbol(r.value.name)

        in_interface = len(expr.get_user_nodes(Interface)) > 0

        args = expr.bind_c_arguments if is_bind_c_function_def else expr.arguments
        results = expr.bind_c_results if is_bind_c_function_def else expr.results

        if in_interface:
            self._get_python_args(args)
            func_args = [FunctionDefArgument(a) for a in self._python_object_map.values()]
            body = []
        else:
            func_args, body = self._unpack_python_args(args)
            func_args = [FunctionDefArgument(a) for a in func_args]

        body += [l for a in args for l in self._wrap(a)]

        call_args = [self.scope.find(self.scope.get_expected_name(a.var.name), category='variables') for a in expr.arguments]
        call_res  = [self.scope.find(self.scope.get_expected_name(r.var.name), category='variables') for a in expr.results]

        result_wrap = [self._wrap(r) for r in results]

        func_results = [FunctionDefResult(r[-1].lhs) for r in result_wrap]

        if call_res:
            body.append(Assign(call_res, FunctionCall(expr, call_args)))
        else:
            body.append(FunctionCall(expr, call_args))

        body += [l for l in result_wrap]

        self.exit_scope()

        if not in_interface:
            res = self.get_new_PyObject("result")
            body.append(AliasAssign(res, PyBuildValueNode(func_results)))
            body.append(Return([res]))
            func_results = [FunctionDefResult(res)]
        elif func_results:
            body.append(Return([func_results]))

        return FunctionDef(func_name, func_args, func_results, body, scope=func_scope)

    def _wrap_FunctionDefArgument(self, expr):

        collect_arg = self._python_object_map[expr]
        in_interface = len(expr.get_user_nodes(Interface)) > 0

        orig_var = getattr(expr, 'original_function_argument_variable', expr.var)
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
            [self.scope.insert_variable(v,s.name) for v,s in zip(shape_vars, expr.shape)]
            [self.scope.insert_variable(v,s.name) for v,s in zip(stride_vars, expr.strides)]

            c_arg = self.scope.find(self.scope.get_expected_name(orig_var.name), category='variables')

            body.append(AliasAssign(arg_var, FunctionCall(array_get_data, [c_arg])))
            body.extend(Assign(s, FunctionCall(array_get_dim, [c_arg, i])) for i,s in enumerate(shape_vars))
            if orig_var.order == 'C':
                body.extend(Assign(s, FunctionCall(array_get_c_step, [c_arg, i])) for i,s in enumerate(stride_vars))
            else:
                body.extend(Assign(s, FunctionCall(array_get_f_step, [c_arg, i])) for i,s in enumerate(stride_vars))

        return body

    def _wrap_FunctionDefResult(self, expr):

        orig_var = expr.original_function_result_variable

        c_res = orig_var.clone(self.scope.get_expected_name(orig_var.name), is_argument = False)

        python_res = self.get_new_PyObject(orig_var.name+'_obj')

        body = [Assign(python_res, FunctionCall(C_to_Python(variable), [ObjectAddress(variable)]))]

        if orig_var.rank:
            body.append(Deallocate(c_res))

        return body

    def _wrap_BindCFunctionDefResult(self, expr):

        orig_var = expr.original_function_result_variable

        c_res = orig_var.clone(self.scope.get_expected_name(orig_var.name), is_argument = False)

        python_res = self.get_new_PyObject(orig_var.name+'_obj')

        if orig_var.rank:
            arg_var = expr.var.clone(self.scope.get_expected_name(expr.var.name), is_argument = False, memory_handling='alias')
            shape_vars = [s.clone(self.scope.get_expected_name(s.name), is_argument = False) for s in expr.shape]
            self.scope.insert_variable(arg_var, expr.var.name)
            [self.scope.insert_variable(v,s.name) for v,s in zip(shape_vars, expr.shape)]

            body.append(Allocate(c_res, shape = shape_vars, order = orig_var.order,
                status='unallocated'))
            body.append(AliasAssign(DottedVariable(NativeVoid(), 'raw_data', memory_handling = 'alias',
                lhs=orig_var), nd_var))

        body = [Assign(python_res, FunctionCall(C_to_Python(c_res), [ObjectAddress(c_res)]))]

        if orig_var.rank:
            body.append(Deallocate(c_res))

        return body
