# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#

from collections import OrderedDict

from pyccel.codegen.printing.ccode import CCodePrinter

from pyccel.ast.bind_c   import BindCPointer
from pyccel.ast.bind_c   import BindCModule, BindCFunctionDef, BindCFunctionDefArgument
from pyccel.ast.bind_c   import BindCFunctionDefResult

from pyccel.ast.builtins import PythonTuple, PythonType

from pyccel.ast.core import Assign, AliasAssign, FunctionDef, FunctionAddress
from pyccel.ast.core import If, IfSection, Return, FunctionCall, Deallocate
from pyccel.ast.core import SeparatorComment, Allocate
from pyccel.ast.core import Import, Module, Declare
from pyccel.ast.core import AugAssign, CodeBlock
from pyccel.ast.core import FunctionDefArgument, FunctionDefResult

from pyccel.ast.cwrapper    import PyArg_ParseTupleNode, PyBuildValueNode
from pyccel.ast.cwrapper    import PyArgKeywords
from pyccel.ast.cwrapper    import Py_None, Py_DECREF
from pyccel.ast.cwrapper    import generate_datatype_error, set_python_error_message
from pyccel.ast.cwrapper    import scalar_object_check
from pyccel.ast.cwrapper    import PyccelPyObject
from pyccel.ast.cwrapper    import C_to_Python, Python_to_C
from pyccel.ast.cwrapper    import PyModule_AddObject

from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeFloat
from pyccel.ast.datatypes import datatype, NativeVoid

from pyccel.ast.literals  import LiteralTrue, LiteralInteger, LiteralString
from pyccel.ast.literals  import Nil

from pyccel.ast.numpy_wrapper   import array_type_check
from pyccel.ast.numpy_wrapper   import pyarray_to_ndarray
from pyccel.ast.numpy_wrapper   import array_get_data, array_get_dim, array_get_c_step, array_get_f_step

from pyccel.ast.operators import PyccelEq, PyccelNot, PyccelOr, PyccelAssociativeParenthesis
from pyccel.ast.operators import PyccelIsNot, PyccelLt, PyccelUnarySub

from pyccel.ast.variable  import Variable, DottedVariable

from pyccel.ast.c_concepts import ObjectAddress

from pyccel.parser.scope  import Scope

from pyccel.errors.errors   import Errors

errors = Errors()

__all__ = ["CWrapperCodePrinter", "cwrappercode"]

dtype_registry = {('pyobject'     , 0) : 'PyObject',
                  ('pyarrayobject', 0) : 'PyArrayObject',
                  ('void'         , 0) : 'void',
                  ('bind_c_ptr'   , 0) : 'void'}

module_imports  = [Import('numpy_version', Module('numpy_version',(),())),
            Import('numpy/arrayobject', Module('numpy/arrayobject',(),())),
            Import('cwrapper', Module('cwrapper',(),()))]

cwrapper_ndarray_import = Import('cwrapper_ndarrays', Module('cwrapper_ndarrays', (), ()))

class CWrapperCodePrinter(CCodePrinter):
    """
    A printer for printing the C-Python interface.

    A printer to convert Pyccel's AST describing a translated module,
    to strings of C code which provide an interface between the module
    and Python code.
    As for all printers the navigation of this file is done via _print_X
    functions.

    Parameters
    ----------
    filename : str
            The name of the file being pyccelised.
    target_language : str
            The language which the code was translated to [fortran/c].
    **settings : dict
            Any additional arguments which are necessary for CCodePrinter.
    """
    def __init__(self, filename, target_language, **settings):
        CCodePrinter.__init__(self, filename, **settings)
        self._target_language = target_language
        self._to_free_PyObject_list = []
        self._function_wrapper_names = dict()
        self._module_name = None


    # --------------------------------------------------------------------
    #                       Helper functions
    # --------------------------------------------------------------------

    def is_c_pointer(self, a):
        """
        Indicate whether the object is a pointer in C code.

        This function extends `CCodePrinter.is_c_pointer` to specify more objects
        which are always accesed via a C pointer.

        Parameters
        ----------
        a : PyccelAstNode
            The object whose storage we are enquiring about.

        Returns
        -------
        bool
            True if a C pointer, False otherwise.

        See Also
        --------
        CCodePrinter.is_c_pointer : The extended function.
        """
        if isinstance(a.dtype, PyccelPyObject):
            return True
        elif isinstance(a, PyBuildValueNode):
            return True
        else:
            return CCodePrinter.is_c_pointer(self,a)

    def get_python_name(self, scope, obj):
        """
        Get the name of object as defined in the original python code.

        Get the name of the object as it was originally defined in the
        Python code being translated. This name may have changed before
        the printing stage in the case of name clashes or language interfaces.

        Parameters
        ----------
        scope : pyccel.parser.scope.Scope
            The scope where the object was defined.

        obj : pyccel.ast.basic.Basic
            The object whose name we wish to identify.

        Returns
        -------
        str
            The original name of the object.
        """
        if isinstance(obj, BindCFunctionDef):
            return scope.get_python_name(obj.original_function.name)
        elif isinstance(obj, BindCModule):
            return obj.original_module.name
        else:
            return scope.get_python_name(obj.name)

    def function_signature(self, expr, print_arg_names = True):
        args = list(expr.arguments)
        if any([isinstance(a.var, FunctionAddress) for a in args]):
            # Functions with function addresses as arguments cannot be
            # exposed to python so there is no need to print their signature
            return ''
        else:
            return CCodePrinter.function_signature(self, expr, print_arg_names)

    def get_declare_type(self, expr):
        """
        Get the string which describes the type in a declaration.

        This function extends `CCodePrinter.get_declare_type` to specify types
        which are only relevant in the C-Python interface.

        Parameters
        ----------
        expr : Variable
            The variable whose type should be described.

        Returns
        -------
        str
            The code describing the type.

        Raises
        ------
        PyccelCodegenError
            If the type is not supported in the C code or the rank is too large.

        See Also
        --------
        CCodePrinter.get_declare_type : The extended function.
        """
        if expr.dtype is BindCPointer():
            return 'void*'
        dtype = self._print(expr.dtype)
        prec  = expr.precision
        if dtype != "pyarrayobject":
            return CCodePrinter.get_declare_type(self, expr)
        else :
            dtype = self.find_in_dtype_registry(dtype, prec)

        if self.is_c_pointer(expr):
            return f'{dtype}*'
        else:
            return dtype

    def find_in_dtype_registry(self, dtype, prec):
        """
        Find the corresponding C dtype in the dtype_registry
        raise PYCCEL_RESTRICTION_TODO if not found

        Parameters
        -----------
        dtype : String
            expression data type

        prec  : Integer
            expression precision

        Returns
        -------
        dtype : String
        """
        try :
            return dtype_registry[(dtype, prec)]
        except KeyError:
            return CCodePrinter.find_in_dtype_registry(self, dtype, prec)

    def get_default_assign(self, arg, func_arg, value):
        """
        Provide the Assign which initialises an argument to its default value.

        When a function def argument has a default value, this function
        provides the code which initialises the argument. This value can
        then either be used or overwritten with the provided argument.

        Parameters
        ----------
        arg : Variable
            The Variable where the default value should be saved.
        func_arg : FunctionDefArgument
            The argument object where the value may be provided.
        value : PyccelAstNode
            The default value which should be assigned.

        Returns
        -------
        Assign
            The code describing the assignement.

        Raises
        ------
        NotImplementedError
            If the type of the default value is not handled.
        """
        if func_arg.is_optional:
            return AliasAssign(arg, Py_None)
        elif isinstance(arg.dtype, (NativeFloat, NativeInteger, NativeBool)):
            return Assign(arg, value)
        elif isinstance(arg.dtype, PyccelPyObject):
            return AliasAssign(arg, Py_None)
        else:
            raise NotImplementedError('Default values are not implemented for this datatype : {}'.format(func_arg.dtype))

    def static_function_signature(self, expr):
        """
        Get the C representation of the function signature using only basic types.

        Extract from the function definition `expr` all the
        information (name, input, output) needed to create the
        function signature used for C/Fortran binding and return
        a string describing the function.

        Parameters
        ----------
        expr : FunctionDef
            The function definition for which a signature is needed.

        Returns
        -------
        str
            Signature of the function.
        """
        #if target_language is C no need for the binding
        if self._target_language == 'c':
            return self.function_signature(expr)

        args = [a.var for a in expr.arguments]
        results = [r.var for r in expr.results]
        if len(results) == 1:
            ret_type = self.get_declare_type(results[0])
        elif len(results) > 1:
            ret_type = self._print(datatype('int'))
            args += [a.clone(name = a.name, memory_handling='alias') for a in results]
        else:
            ret_type = self._print(datatype('void'))
        name = expr.name
        if not args:
            arg_code = 'void'
        else:
            arg_code = ', '.join(self.function_signature(i, False)
                        if isinstance(i, FunctionAddress)
                        else self.get_static_declare_type(i)
                        for i in args)

        return f'{ret_type} {name}({arg_code})'

    def get_static_args(self, argument):
        """
        Get the value(s) which should be passed for the provided argument.

        Get the value(s) which should be passed to the function for the
        specified argument. In the case of a BindCFunctionDef (when the
        target language is Fortran) and an argument with rank > 0,
        multiple arguments are needed:
        - buffer holding data.
        - shape of array in each dimension.
        - stride for the array in each dimension.

        Parameters
        ----------
        argument : Variable
            Variable holding information needed (rank).

        Returns
        -------
        List of arguments
            List that can contain Variables and FunctionCalls.

        Examples
        --------
        If target language is Fortran:
        >>> x = Variable('int', 'x', rank=2, order='c')
        >>> self.get_static_args(x)
        [&nd_data(x), nd_ndim(x, 0), nd_ndim(x, 1), nd_nstep_C(x, 0), nd_nstep_C(x, 1)]

        If target language is C:
        >>> x = Variable('int', 'x', rank=2, order='c')
        >>> self.get_static_args(x)
        [x]
        """

        if self._target_language == 'fortran' and argument.rank > 0:
            arg_address = ObjectAddress(argument)
            static_args = [ObjectAddress(FunctionCall(array_get_data, [arg_address]))]
            static_args+= [
                FunctionCall(array_get_dim, [arg_address, i]) for i in range(argument.rank)
            ]
            if argument.order == 'C':
                static_args+= [
                    FunctionCall(array_get_c_step, [arg_address, i]) for i in range(argument.rank)
                ]
            else:
                static_args+= [
                    FunctionCall(array_get_f_step, [arg_address, i]) for i in range(argument.rank)
                ]
        else:
            static_args = [argument]

        return static_args

    def get_static_declare_type(self, variable):
        """
        Get the declaration type of a variable which may be passed to a Fortran binding function.

        Get the declaration type of a variable, this function is used for
        C/Fortran binding using native C datatypes.

        Parameters
        ----------
        variable : Variable
            Variable holding information needed to choose the declaration type.

        Returns
        -------
        str
            The code describing the type.
        """
        dtype = self._print(variable.dtype)
        prec  = variable.precision

        dtype = self.find_in_dtype_registry(dtype, prec)

        if self.is_c_pointer(variable):
            return f'{dtype}*'

        else:
            return dtype

    def get_static_results(self, result):
        """
        Get the value(s) which should be used to collect the provided result.

        Get the value(s) which should be collected from the function for the
        specified argument. In the case of a BindCFunctionDefResult (when the
        target language is Fortran) and an argument with rank > 0,
        multiple outputs are needed:
        - buffer holding data.
        - shape of array in each dimension.
        These must then be used to create the expected result variable.

        Parameters
        ----------
        result : FunctionDefResult
            The result of the function which we want to collect.

        Returns
        -------
        body : list
            Additional instructions (allocations and pointer assignments) for function body.

        static_results : list
            Expanded list of function arguments corresponding to the given result.

        Examples
        --------
        If target language is Fortran:
        >>> x_res = BindCFunctionDefResult(Variable(BindCPointer(), 'x_ptr', memory_handling='alias'), Variable('int', 'x', rank=2, order='c'), scope)
        >>> self.get_static_results(x)
        [allocate(x, [x_shape_0, x_shape_1]), x.raw_data=>x_ptr], [&nd_data(x_ptr), x_shape_0, x_shape_1]

        If target language is C:
        >>> x_res = FunctionDefResult(Variable('int', 'x', rank=2, order='c'))
        >>> self.get_static_args(x)
        [], [Variable(x)]
        """

        body = []
        var_name = self.scope.get_expected_name(result.var.name)

        if isinstance(result, BindCFunctionDefResult) and result.shape:
            shape = [self.scope.get_temporary_variable(s) for s in result.shape]
            orig_name = result.original_function_result_variable.name
            orig_var = self.scope.find(self.scope.get_expected_name(orig_name), category='variables')
            nd_var = self.scope.find(var_name, category='variables')
            body.append(Allocate(orig_var, shape = shape, order = orig_var.order,
                status='unallocated'))
            body.append(AliasAssign(DottedVariable(NativeVoid(), 'raw_data', memory_handling = 'alias',
                lhs=orig_var), nd_var))

            static_results = [ObjectAddress(nd_var), *shape]

        else:
            var = self.scope.find(var_name, category='variables')
            static_results = [var]

        return body, static_results

    def _get_static_func_call_code(self, expr, static_func_args, results):
        """
        Get all code necessary to call the wrapped function.

        Get the function call which calls the underlying translated function
        being wrapped. This may involve creating new variables in order to
        call the function in a compatible way.

        Parameters
        ----------
        expr : FunctionDef
            The function being wrapped.

        static_func_args : List of arguments
            Arguments compatible with the static function.

        results : List of results
            Results of the wrapped function.

        Returns
        -------
        list of pyccel.ast.basic.Basic
            List of nodes describing the instructions which call the
            wrapped function.
        """
        body = []
        if len(results) == 0:
            body.append(FunctionCall(expr, static_func_args))
        else:
            static_func_results = []
            for r in results:
                b, s = self.get_static_results(r)
                body.extend(b)
                static_func_results.extend(s)

            results   = static_func_results if len(static_func_results)>1 else static_func_results[0]
            func_call = Assign(results,FunctionCall(expr, static_func_args))
            body.insert(0, func_call)

        return body

    def _handle_is_operator(self, Op, expr):
        if expr.rhs is Py_None:
            lhs = ObjectAddress(expr.lhs)
            rhs = ObjectAddress(expr.rhs)
            lhs = self._print(lhs)
            rhs = self._print(rhs)
            return '{} {} {}'.format(lhs, Op, rhs)
        else:
            return super()._handle_is_operator(Op, expr)

    # -------------------------------------------------------------------
    # Functions managing the creation of wrapper body
    # -------------------------------------------------------------------

    def insert_constant(self, mod_name, var_name, var, collect_var):
        """
        Insert a variable into the module

        Parameters
        ----------
        mod_name    : str
                      The name of the module variable
        var_name    : str
                      The name which will be used to identify the
                      variable in python. (This is usually var.name,
                      however it may differ due to the case-insensitivity
                      of fortran)
        var         : Variable
                      The module variable
        collect_var : Variable
                      A PyObject* variable to store temporaries

        Returns
        -------
        list : A list of PyccelAstNodes to be appended to the body of the
                pymodule initialisation function
        """
        if var.rank != 0:
            self.add_import(cwrapper_ndarray_import)

        collect_value = AliasAssign(collect_var,
                                FunctionCall(C_to_Python(var), [ObjectAddress(var)]))
        add_expr = PyModule_AddObject(mod_name, var_name, collect_var)
        if_expr = If(IfSection(PyccelLt(add_expr,LiteralInteger(0)),
                        [FunctionCall(Py_DECREF, [collect_var]),
                         Return([PyccelUnarySub(LiteralInteger(1))])]))
        return [collect_value, if_expr]

    def get_module_exec_function(self, expr, exec_func_name):
        """
        Create code which initialises a module.

        Create the function which executes any statements which happen
        when the module is loaded.

        Parameters
        ----------
        expr : Module
            The module being wrapped.

        exec_func_name : str
            The name of the function.

        Returns
        -------
        str
            The code for a function which initialises a module.
        """
        # Create scope for the module initialisation function
        scope = self.scope.new_child_scope(exec_func_name)
        self.set_scope(scope)

        #Create module variable
        mod_var_name = self.scope.get_new_name('m')
        mod_var = Variable(dtype = PyccelPyObject(),
                      name       = mod_var_name,
                      memory_handling = 'alias')
        scope.insert_variable(mod_var)

        # Collect module variables from translated code
        body = []
        if isinstance(expr, BindCModule):
            orig_vars_to_wrap = [v for v in expr.original_module.variables if not v.is_private]
            wrapper_funcs = {f.original_function: f for f in expr.variable_wrappers}
            # Collect python compatible module variables
            vars_to_wrap = []
            for v in orig_vars_to_wrap:
                if v in wrapper_funcs:
                    # Get pointer to store array data
                    var = scope.get_temporary_variable(dtype_or_var = v,
                            name = v.name,
                            memory_handling = 'alias',
                            rank = 0, shape = None, order = None)
                    # Create variables to store the shape of the array
                    shape = [scope.get_temporary_variable(NativeInteger(),
                            v.name+'_size') for _ in range(v.rank)]
                    # Get the bind_c function which wraps a fortran array and returns c objects
                    var_wrapper = wrapper_funcs[v]
                    # Call bind_c function
                    call = Assign(PythonTuple(ObjectAddress(var), *shape), FunctionCall(var_wrapper, ()))
                    body.append(call)

                    # Create ndarray to store array data
                    nd_var = self.scope.get_temporary_variable(dtype_or_var = v,
                            name = v.name,
                            memory_handling = 'alias'
                            )
                    alloc = Allocate(nd_var, shape=shape, order=nd_var.order, status='unallocated')
                    body.append(alloc)
                    # Save raw_data into ndarray to obtain useable pointer
                    set_data = AliasAssign(DottedVariable(NativeVoid(), 'raw_data',
                            memory_handling = 'alias', lhs=nd_var), var)
                    body.append(set_data)
                    # Save the ndarray to vars_to_wrap to be handled as if it came from C
                    vars_to_wrap.append(nd_var)
                else:
                    # Ensure correct name
                    w = v.clone(scope.get_expected_name(v.name.lower()))
                    assign = v.get_user_nodes(Assign)[0]
                    # assign.fst should always exist, but is not always set when the
                    # Assign is created in the codegen stage
                    if assign.fst:
                        w.set_fst(assign.fst)
                    vars_to_wrap.append(w)
        else:
            orig_vars_to_wrap = [v for v in expr.variables if not v.is_private]
            vars_to_wrap = orig_vars_to_wrap
        var_names = [str(self.get_python_name(expr.scope, v)) for v in orig_vars_to_wrap]

        # If there are any variables in the module then add them to the module object
        if vars_to_wrap:
            # Create variable for temporary python objects
            tmp_var_name = self.scope.get_new_name('tmp')
            tmp_var = Variable(dtype = PyccelPyObject(),
                          name       = tmp_var_name,
                          memory_handling = 'alias')
            scope.insert_variable(tmp_var)
            # Add code to add variable to module
            body.extend(l for n,v in zip(var_names,vars_to_wrap) for l in self.insert_constant(mod_var_name, n, v, tmp_var))

        if expr.init_func:
            # Call init function code
            body.insert(0,FunctionCall(expr.init_func,[],[]))

        body.append(Return([LiteralInteger(0)]))
        self.exit_scope()

        func = FunctionDef(name = exec_func_name,
            arguments = (FunctionDefArgument(mod_var),),
            results = (FunctionDefResult(scope.get_temporary_variable(NativeInteger(),
                precision = 4)),),
            body = CodeBlock(body),
            scope = scope)
        func_code = super()._print_FunctionDef(func).split('\n')
        func_code[1] = "static "+func_code[1]


        return '\n'.join(func_code)

    def _print_BindCPointer(self, expr):
        return 'bind_c_ptr'

    #--------------------------------------------------------------------
    #                 _print_ClassName functions
    #--------------------------------------------------------------------

    def _print_DottedName(self, expr):
        names = expr.name
        return '.'.join(self._print(n) for n in names)

    def _print_PyInterface(self, expr):
        funcs_to_print = (*expr.functions, expr.type_check_func, expr.interface_func)
        return '\n'.join(self._print(f) for f in funcs_to_print)

    def _print_PyccelPyObject(self, expr):
        return 'pyobject'

    def _print_PyArg_ParseTupleNode(self, expr):
        name    = 'PyArg_ParseTupleAndKeywords'
        pyarg   = expr.pyarg
        pykwarg = expr.pykwarg
        flags   = expr.flags
        # All args are modified so even pointers are passed by address
        args    = ', '.join(['&{}'.format(a.name) for a in expr.args])

        if expr.args:
            code = '{name}({pyarg}, {pykwarg}, "{flags}", {kwlist}, {args})'.format(
                            name=name,
                            pyarg=pyarg,
                            pykwarg=pykwarg,
                            flags = flags,
                            kwlist = expr.arg_names.name,
                            args = args)
        else :
            code ='{name}({pyarg}, {pykwarg}, "", {kwlist})'.format(
                    name=name,
                    pyarg=pyarg,
                    pykwarg=pykwarg,
                    kwlist = expr.arg_names.name)

        return code

    def _print_PyBuildValueNode(self, expr):
        name  = 'Py_BuildValue'
        flags = expr.flags
        args  = ', '.join(['{}'.format(self._print(a)) for a in expr.args])
        #to change for args rank 1 +
        if expr.args:
            code = f'(*{name}("{flags}", {args}))'
        else :
            code = f'(*{name}(""))'
        return code

    def _print_PyArgKeywords(self, expr):
        arg_names = ',\n'.join(['"{}"'.format(a) for a in expr.arg_names] + [self._print(Nil())])
        return ('static char *{name}[] = {{\n'
                        '{arg_names}\n'
                        '}};\n'.format(name=expr.name, arg_names = arg_names))

    def _print_PyModule_AddObject(self, expr):
        return 'PyModule_AddObject({}, {}, {})'.format(
                expr.mod_name,
                self._print(expr.name),
                self._print(expr.variable))

    def _print_PyModule(self, expr):
        scope = Scope(original_symbols = expr.scope.python_names.copy())
        self.set_scope(scope)
        # The initialisation and deallocation shouldn't be exposed to python
        funcs_to_wrap = [f for f in expr.funcs if f not in (expr.init_func, expr.free_func)]

        # Insert declared objects into scope
        variables = expr.original_module.variables if isinstance(expr, BindCModule) else expr.variables
        for f in expr.funcs:
            scope.insert_symbol(f.name.lower())
        for v in variables:
            if not v.is_private:
                scope.insert_symbol(v.name.lower())

        funcs = []
        #if self._target_language == 'fortran':
        #    vars_to_wrap_decs = [Declare(v.dtype, v.clone(v.name.lower()), module_variable=True) \
        #                            for v in variables if not v.is_private and v.rank == 0]

        #    for f in expr.original_module.funcs:
        #        if f.is_private:
        #            funcs.append(f)
        #else:
        #    vars_to_wrap_decs = [Declare(v.dtype, v, module_variable=True) \
        #                            for v in expr.variables if not v.is_private]

        self._module_name  = self.get_python_name(scope, expr)
        sep = self._print(SeparatorComment(40))

        function_signatures = ''.join(f'{self.static_function_signature(f)};\n' for f in expr.external_funcs)
        #if self._target_language == 'fortran':
        #    function_signatures += ''.join(f'{self.static_function_signature(f)};\n' for f in expr.variable_wrappers)

        interface_funcs = [f.name for i in expr.interfaces for f in i.functions]
        funcs += [*expr.interfaces, *(f for f in funcs_to_wrap if f.name not in interface_funcs)]

        self._in_header = True
        #decs = ''.join('extern '+self._print(d) for d in vars_to_wrap_decs)
        self._in_header = False

        function_defs = '\n'.join(self._print(f) for f in funcs)
        method_def_func = ''.join(('{{\n'
                                     '"{name}",\n'
                                     '(PyCFunction){wrapper_name},\n'
                                     'METH_VARARGS | METH_KEYWORDS,\n'
                                     '{doc_string}\n'
                                     '}},\n').format(
                                            name = self.get_python_name(expr.scope, f.original_function),
                                            wrapper_name = f.name,
                                            doc_string = self._print(LiteralString('\n'.join(f.doc_string.comments))) \
                                                        if f.doc_string else '""')
                                     for f in funcs if f is not expr.init_func)

        slots_name = self.scope.get_new_name('{}_slots'.format(expr.name))
        exec_func_name = self.scope.get_new_name('exec_func')
        slots_def = ('static PyModuleDef_Slot {name}[] = {{\n'
                     '{{Py_mod_exec, {exec_func}}},\n'
                     '{{0, NULL}},\n'
                     '}};\n').format(name = slots_name,
                             exec_func = exec_func_name)

        method_def_name = self.scope.get_new_name('{}_methods'.format(expr.name))
        method_def = ('static PyMethodDef {method_def_name}[] = {{\n'
                        '{method_def_func}'
                        '{{ NULL, NULL, 0, NULL}}\n'
                        '}};\n'.format(method_def_name = method_def_name,
                            method_def_func = method_def_func
                            ))

        module_def_name = self.scope.get_new_name('{}_module'.format(expr.name))
        module_def = ('static struct PyModuleDef {module_def_name} = {{\n'
                'PyModuleDef_HEAD_INIT,\n'
                '/* name of module */\n'
                '\"{mod_name}\",\n'
                '/* module documentation, may be NULL */\n'
                'NULL,\n' #TODO: Add documentation
                '/* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */\n'
                '0,\n'
                '{method_def_name},\n'
                '{slots_name}\n'
                '}};\n'.format(module_def_name = module_def_name,
                    mod_name = self._module_name,
                    method_def_name = method_def_name,
                    slots_name = slots_name))

        exec_func = self.get_module_exec_function(expr, exec_func_name)

        init_func = ('PyMODINIT_FUNC PyInit_{mod_name}(void)\n{{\n'
                'import_array();\n'
                'return PyModuleDef_Init(&{module_def_name});\n'
                '}}\n'.format(mod_name=self._module_name,
                    module_def_name = module_def_name))

        # Print imports last to be sure that all additional_imports have been collected
        for i in expr.imports:
            self.add_import(i)
        imports  = module_imports.copy()
        imports += self._additional_imports.values()
        imports  = ''.join(self._print(i) for i in imports)

        self.exit_scope()

        return ('#define PY_ARRAY_UNIQUE_SYMBOL CWRAPPER_ARRAY_API\n'
                '{imports}\n'
                #'{variable_declarations}\n'
                '{function_signatures}\n'
                '{sep}\n'
                '{sep}\n'
                '{function_defs}\n'
                '{exec_func}\n'
                '{sep}\n'
                '{method_def}\n'
                '{sep}\n'
                '{slots_def}\n'
                '{sep}\n'
                '{module_def}\n'
                '{sep}\n'
                '{init_func}'.format(
                    imports = imports,
                    #variable_declarations = decs,
                    function_signatures = function_signatures,
                    sep = sep,
                    function_defs = function_defs,
                    exec_func = exec_func,
                    method_def = method_def,
                    slots_def  = slots_def,
                    module_def = module_def,
                    init_func = init_func))

def cwrappercode(expr, filename, target_language, assign_to=None, **settings):
    """Converts an expr to a string of c wrapper code

    expr : Expr
        A pyccel expression to be converted.
    filename : str
        The name of the file being translated. Used in error printing
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, cfunction_string)]. See below
        for examples.
    dereference : iterable, optional
        An iterable of symbols that should be dereferenced in the printed code
        expression. These would be values passed by address to the function.
        For example, if ``dereference=[a]``, the resulting code would print
        ``(*a)`` instead of ``a``.
    """

    return CWrapperCodePrinter(filename, target_language, **settings).doprint(expr, assign_to)
