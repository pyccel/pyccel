# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=no-self-use

from collections import OrderedDict

from pyccel.codegen.printing.ccode import CCodePrinter

from pyccel.ast.bind_c   import as_static_function, wrap_module_array_var, BindCPointer

from pyccel.ast.builtins import PythonTuple, PythonType

from pyccel.ast.core import Assign, AliasAssign, FunctionDef, FunctionAddress
from pyccel.ast.core import If, IfSection, Return, FunctionCall, Deallocate
from pyccel.ast.core import SeparatorComment, Allocate
from pyccel.ast.core import Import, Module, Declare
from pyccel.ast.core import AugAssign, CodeBlock

from pyccel.ast.cwrapper    import PyArg_ParseTupleNode, PyBuildValueNode
from pyccel.ast.cwrapper    import PyArgKeywords
from pyccel.ast.cwrapper    import Py_None, Py_DECREF
from pyccel.ast.cwrapper    import generate_datatype_error, PyErr_SetString
from pyccel.ast.cwrapper    import scalar_object_check, flags_registry
from pyccel.ast.cwrapper    import PyccelPyObject
from pyccel.ast.cwrapper    import C_to_Python, Python_to_C
from pyccel.ast.cwrapper    import PyModule_AddObject

from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeFloat
from pyccel.ast.datatypes import datatype, NativeVoid

from pyccel.ast.literals  import LiteralTrue, LiteralInteger, LiteralString
from pyccel.ast.literals  import Nil

from pyccel.ast.numpy_wrapper   import array_type_check
from pyccel.ast.numpy_wrapper   import pyarray_to_ndarray
from pyccel.ast.numpy_wrapper   import array_get_data, array_get_dim

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
                  ('bind_c_ptr'   , 0) : 'void*'}

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
        self._cast_functions_dict = OrderedDict()
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

    def get_new_PyObject(self, name):
        """
        Create new PyccelPyObject Variable with the desired name

        Parameters
        -----------
        name       : String
            The desired name

        Returns: Variable
        -------
        """
        return Variable(dtype=PyccelPyObject(),
                        name=self.scope.get_new_name(name),
                        memory_handling='alias')

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

    def get_static_function(self, function):
        """

        Create a static FunctionDef from the argument used for
        C/fortran binding.
        If target language is C return the argument function
        If target language is Fortran, return a static function which
        takes both the data and the shapes as arguments

        Parameters
        ----------
        function    : FunctionDef
            FunctionDef holding information needed to create static function

        Returns   :
        -----------
        static_func : FunctionDef
        """
        if self._target_language == 'fortran':
            static_func = as_static_function(function, mod_scope = self.scope)
        else:
            static_func = function

        return static_func

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
        if len(expr.results) == 1:
            ret_type = self.get_declare_type(expr.results[0])
        elif len(expr.results) > 1:
            ret_type = self._print(datatype('int'))
            args += [a.clone(name = a.name, memory_handling='alias') for a in expr.results]
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
        Create bind_C arguments for arguments rank > 0 in fortran.
        needed in static function call
        func(a) ==> static_func(nd_dim(a) , nd_data(a))
        where nd_data(a) = buffer holding data
              nd_dim(a)  = size of array

        Parameters
        ----------
        argument    : Variable
            Variable holding information needed (rank)

        Returns     : List of arguments
            List that can contains Variables and FunctionCalls
        -----------
        """

        if self._target_language == 'fortran' and argument.rank > 0:
            arg_address = ObjectAddress(argument)
            static_args = [
                FunctionCall(array_get_dim, [arg_address, i]) for i in range(argument.rank)
            ]
            static_args.append(ObjectAddress(FunctionCall(array_get_data, [arg_address])))
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
            return '{0}*'.format(dtype)

        elif self._target_language == 'fortran' and variable.rank > 0:
            return '{0}*'.format(dtype)

        else:
            return '{0}'.format(dtype)

    def get_static_results(self, result):
        """
        Create bind_C results for results rank > 0 in fortran.
        needed in static function call
        func(a) ==> static_func(a.shape[0] , &a->raw_data)

        Parameters
        ----------
        result      : Variable
            Variable holding information needed (rank)

        Returns
        -------
        body : list
            Additional instructions (allocations and pointer assignments) for function body

        static_results : list
            Expanded list of function arguments corresponding to the given result
        """

        body = []

        if self._target_language == 'fortran' and result.rank > 0:
            sizes = [
                     self.scope.get_temporary_variable(NativeInteger(),
                         result.name+'_size') for _ in range(result.rank)
                     ]
            nd_var = self.scope.get_temporary_variable(dtype_or_var = NativeVoid(),
                    name = result.name,
                    memory_handling = 'alias')
            body.append(Allocate(result, shape = sizes, order = result.order,
                status='unallocated'))
            body.append(AliasAssign(DottedVariable(NativeVoid(), 'raw_data', memory_handling = 'alias',
                lhs=result), nd_var))

            static_results = [ObjectAddress(nd_var), *sizes]

        else:
            static_results = [result]

        return body, static_results

    def _get_static_func_call_code(self, expr, static_func_args, results):
        """
        Get all code necessary to call the wrapped function

        Parameters
        ----------
        expr             : FunctionDef
                           The function being wrapped
        static_func_args : List of arguments
                           Arguments compatible with the static function
        results          : List of results
                           Results of the wrapped function

        Returns
        -------
        body             : List of Basic Nodes
                           List of nodes describing the instructions which call the
                           wrapped function

        """
        body = []
        static_function = self.get_static_function(expr)
        if len(results) == 0:
            body.append(FunctionCall(static_function, static_func_args))
        else:
            static_func_results = []
            for r in results:
                b, s = self.get_static_results(r)
                body.extend(b)
                static_func_results.extend(s)

            results   = static_func_results if len(static_func_results)>1 else static_func_results[0]
            func_call = Assign(results,FunctionCall(static_function, static_func_args))
            body.insert(0, func_call)

        return body

    def _get_check_type_statement(self, variable, collect_var, compulsory):
        """
        Check if the provided variable has the expected type.

        Get the code which checks if the variable collected from python
        has the expected type.

        Parameters
        ----------
        variable : Variable
                      The variable containing the PythonObject.
        collect_var : Variable
                      The variable in which the result will be saved,
                      used to provide information about the expected type.
        compulsory : bool
                      Indicates whether the argument is a compulsory argument
                      to the function (if not then it must have a default or
                      be optional).

        Returns
        -------
        str
                A string containing the code which determines whether 'variable'
                contains an object which can be saved in 'collect_var'.
        """

        if variable.rank > 0 :
            check = array_type_check(collect_var, variable, False)

        else :
            check = scalar_object_check(collect_var, variable)

        if not compulsory:
            default = PyccelEq(ObjectAddress(collect_var), ObjectAddress(Py_None))
            check = PyccelAssociativeParenthesis(PyccelOr(default, check))

        return check

    def _get_wrapper_name(self, func):
        """
        create wrapper function name

        Parameters
        -----------
        func      : FunctionDef or Interface

        Returns
        -------
        wrapper_name : string
        """
        name         = func.name
        wrapper_name = self.scope.get_new_name(name+"_wrapper")

        self._function_wrapper_names[func.name] = wrapper_name

        return wrapper_name

    def get_wrapper_arguments(self):
        """
        Create wrapper arguments

        Returns
        -------
        List of variables
        """
        python_func_args    = self.get_new_PyObject("args")
        python_func_kwargs  = self.get_new_PyObject("kwargs")
        python_func_selfarg = self.get_new_PyObject("self"  )
        self.scope.insert_variable(python_func_args)
        self.scope.insert_variable(python_func_kwargs)
        self.scope.insert_variable(python_func_selfarg)

        return [python_func_selfarg, python_func_args, python_func_kwargs]


    # -------------------------------------------------------------------
    # Functions managing the creation of wrapper body
    # -------------------------------------------------------------------

    def _valued_variable_management(self, variable, collect_var, tmp_variable, default_value):
        """
        Responsible for creating the body collecting the default value of a Variable
        and the check needed.
        If the Variable is optional create body to collect the new value

        Parameters
        ----------
        variable      : Variable
                        The optional variable
        collect_var   : Variable
                        variable which holds the value collected with PyArg_Parsetuple
        tmp_variable  : Variable
                        The temporary variable  to hold result
        default_value : PyccelAstNode
                        Object containing the default value of the variable

        Returns
        -------
        section      :
            IfSection
        collect_body : List
            list containing the lines necessary to collect the new optional variable value
        """

        valued_var_check  = PyccelEq(ObjectAddress(collect_var), ObjectAddress(Py_None))
        collect_body      = []

        if variable.is_optional:
            collect_body  = [AliasAssign(variable, tmp_variable)]
            section       = IfSection(valued_var_check, [AliasAssign(variable, Nil())])

        else:
            section       = IfSection(valued_var_check, [Assign(variable, default_value)])

        return section, collect_body


    def _body_scalar(self, variable, collect_var, default_value = None, error_check = False, tmp_variable = None):
        """
        Responsible for collecting value and managing error and create the body
        of arguments in format:

        Parameters
        ----------
        variable      : Variable
                        the variable to be collected
        collect_var   : Variable
                        variable which holds the value collected with PyArg_Parsetuple
        default_value : PyccelAstNode
                        Object containing the default value of the variable
                        Default: None
        error_check   : boolean
                        True if checking the data type and raising error is needed
                        Default: False
        tmp_variable  : Variable
                        temporary variable to hold value
                        Default: None

        Returns
        -------
        body : If block
        """

        var      = tmp_variable if tmp_variable else variable
        sections = []

        collect_value = [Assign(var, FunctionCall(Python_to_C(var), [collect_var]))]

        if default_value is not None:
            section, optional_collect = self._valued_variable_management(variable, collect_var, tmp_variable, default_value)
            sections.append(section)
            collect_value += optional_collect

        if error_check:
            check_type = scalar_object_check(collect_var, var)
            sections.append(IfSection(check_type, collect_value))
            error = generate_datatype_error(var)
            sections.append(IfSection(LiteralTrue(), [error, Return([Nil()])]))
        else:
            sections.append(IfSection(LiteralTrue(), collect_value))

        if len(sections)==1 and sections[0].condition == LiteralTrue():
            return sections[0].body
        else:
            return If(*sections)

    def _body_array(self, variable, collect_var, check_type = False) :
        """
        Create the code to extract an array.

        This function is responsible for collecting the value of the array from
        a provided Python variable, and saving it into a C object.
        It also manages potential errors such as if the wrong type is provided.
        Finally it also handles the case of an optional array.

        Parameters
        ----------
        variable : Variable
            The C variable where the result will be stored.
        collect_var : Variable
            The Variable containing the Python object, of type PyObject.
        check_type : bool
            True if the type is needed.

        Returns
        -------
        list
            A list of code statements.
        """
        self.add_import(cwrapper_ndarray_import)
        body = []

        in_if = False

        #check optional :
        if variable.is_optional :
            check = PyccelEq(ObjectAddress(collect_var), ObjectAddress(Py_None))
            body += [IfSection(check, [AliasAssign(variable, Nil())])]
            in_if = True

        if check_type:
            check = array_type_check(collect_var, variable, True)
            body += [IfSection(PyccelNot(check), [Return([Nil()])])]
            in_if = True

        collect_func = FunctionCall(pyarray_to_ndarray, [collect_var])
        if in_if:
            # Use this if other array storage (e.g. cuda arrays) is available
            #body += [IfSection(FunctionCall(PyArray_Check, [collect_var]), [Assign(variable,
            #                    collect_func)])]
            body += [IfSection(LiteralTrue(), [Assign(variable, collect_func)])]
            body = [If(*body)]
        else:
            body = [Assign(variable, collect_func)]

        return body

    def _body_management(self, variable, collect_var, default_value = None, check_type = False):
        """
        Responsible for calling functions that take care of body creation

        Parameters
        ----------
        variable      : Variable
                        The Variable (which may be optional)
        collect_var   : Variable
                        the pyobject type variable  holder of value
        default_value : PyccelAstNode
                        Object containing the default value of the variable
                        Default: None
        check_type    : Boolean
                        True if the type is needed
                        Default: False

        Returns
        -------
        body : list
            A list of statements
        tmp_variable : Variable
            temporary variable to hold value default None
        """
        tmp_variable = None
        body         = []

        if variable.rank > 0:
            body = self._body_array(variable, collect_var, check_type)

        else:
            if variable.is_optional:
                tmp_variable = Variable(dtype=variable.dtype, precision = variable.precision,
                                        name = self.scope.get_new_name(variable.name+"_tmp"))
                self.scope.insert_variable(tmp_variable)

            body = [self._body_scalar(variable, collect_var, default_value, check_type, tmp_variable)]

        return body, tmp_variable

    def untranslatable_function(self, wrapper_name, wrapper_args, wrapper_results, error_msg):
        """
        Certain functions are not handled in the wrapper (e.g. private),
        This creates a wrapper function which raises NotImplementedError
        exception and returns NULL

        Parameters
        ----------
        wrapper_name    : string
            The name of the C wrapper function

        wrapper_args    : list of Variables
            List of variables with dtype PyObject which hold the arguments
            passed to the function

        wrapper_results : Variable
            List containing one variable with dtype PyObject which represents
            the variable which will be returned by the function

        error_msg       : string
            The message to be raised in the NotImplementedError

        Returns
        -------
        code : string
            returns the string containing the printed FunctionDef
        """
        current_scope = self.scope
        wrapper_func = FunctionDef(
                name      = wrapper_name,
                arguments = wrapper_args,
                results   = wrapper_results,
                body      = [
                                PyErr_SetString('PyExc_NotImplementedError',
                                            '"{}"'.format(error_msg)),
                                AliasAssign(wrapper_results[0], Nil()),
                                Return(wrapper_results)
                            ],
                scope     = Scope())

        code = CCodePrinter._print_FunctionDef(self, wrapper_func)
        self.set_scope(current_scope)
        return code

    # -------------------------------------------------------------------
    # Parsing arguments and building values Types functions
    # -------------------------------------------------------------------
    def get_PyArgParseType(self, variable):
        """
        Get the variable which collects the result of PyArgParse.

        This function is responsible for creating any necessary intermediate variables which are used
        to collect the result of PyArgParse.

        Parameters
        ----------
        variable : Variable
            The variable which will be passed to the translated function.

        Returns
        -------
        Variable
            The variable which will be used to collect the argument.
        """

        collect_type = PyccelPyObject()
        collect_var  = Variable(dtype = collect_type,
                            memory_handling='alias',
                            name=self.scope.get_new_name(variable.name+"_tmp"))
        self.scope.insert_variable(collect_var)

        return collect_var

    def get_PyBuildValue(self, variable):
        """
        Responsible for collecting the variable required to build the result
        and the necessary cast function

        Parameters
        ----------
        variable : Variable
            The variable returned by the translated function

        Returns
        -------
        collect_var : Variable
            The variable which will be provided to PyBuild

        cast_func_stmts : functionCall
            call to cast function responsible for the conversion of one data type into another
        """
        if variable.rank != 0:
            self.add_import(cwrapper_ndarray_import)


        cast_function = FunctionCall(C_to_Python(variable), [ObjectAddress(variable)])

        collect_type = PyccelPyObject()
        collect_var = Variable(dtype = collect_type, memory_handling='alias',
            name = self.scope.get_new_name(variable.name+"_tmp"))
        self.scope.insert_variable(collect_var)
        self._to_free_PyObject_list.append(collect_var) #TODO remove in next PR

        return collect_var, cast_function

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
        Create the function which executes any statements which happen
        when the module is loaded

        Parameters
        ----------
        expr           : Module
                         The module being wrapped
        exec_func_name : str
                         The name of the function

        Result
        ------
        str
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
        orig_vars_to_wrap = [v for v in expr.variables if not v.is_private]
        body = []
        if self._target_language == 'fortran':
            # Collect python compatible module variables
            vars_to_wrap = []
            for v in orig_vars_to_wrap:
                if v.rank > 0:
                    # Get pointer to store array data
                    var = scope.get_temporary_variable(dtype_or_var = v,
                            name = v.name,
                            memory_handling = 'alias',
                            rank = 0, shape = None, order = None)
                    # Create variables to store sizes of array
                    sizes = [scope.get_temporary_variable(NativeInteger(),
                            v.name+'_size') for _ in range(v.rank)]
                    # Get the bind_c function which wraps a fortran array and returns c objects
                    var_wrapper = wrap_module_array_var(v, scope, expr)
                    # Call bind_c function
                    call = Assign(PythonTuple(ObjectAddress(var), *sizes), FunctionCall(var_wrapper, ()))
                    body.append(call)

                    # Create ndarray to store array data
                    nd_var = self.scope.get_temporary_variable(dtype_or_var = v,
                            name = v.name,
                            memory_handling = 'alias'
                            )
                    alloc = Allocate(nd_var, shape=sizes, order=nd_var.order, status='unallocated')
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
            vars_to_wrap = orig_vars_to_wrap
        var_names = [str(expr.scope.get_python_name(v.name)) for v in orig_vars_to_wrap]

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
            static_function = self.get_static_function(expr.init_func)
            body.insert(0,FunctionCall(static_function,[],[]))

        body.append(Return([LiteralInteger(0)]))
        self.exit_scope()

        func = FunctionDef(name = exec_func_name,
            arguments = (mod_var,),
            results = (scope.get_temporary_variable(NativeInteger(),
                precision = 4),),
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

    def _print_Interface(self, expr):

        # Find a name for the wrapper function
        wrapper_name = self._get_wrapper_name(expr)

        mod_scope = self.scope
        scope = self.scope.new_child_scope(wrapper_name)
        self.set_scope(scope)

        # Collecting all functions
        funcs = expr.functions

        # Save all used names
        for n in funcs:
            self.scope.insert_symbol(n.name)

        # Collect arguments and results
        wrapper_args    = self.get_wrapper_arguments()
        wrapper_results = [self.get_new_PyObject("result")]
        self.scope.insert_variable(wrapper_results[0])

        # Collect argument names for PyArgParse
        arg_names         = [a.name for a in funcs[0].arguments]
        keyword_list_name = self.scope.get_new_name('kwlist')
        keyword_list      = PyArgKeywords(keyword_list_name, arg_names)
        wrapper_body      = [keyword_list]

        wrapper_body_translations = []
        body_tmp = []

        # To store the mini function responsible for collecting value and calling interfaces functions and return the builded value
        funcs_def = []
        default_value = {} # dict to collect all initialisation needed in the wrapper
        check_var = Variable(dtype = NativeInteger(), name = self.scope.get_new_name("check"))
        scope.insert_variable(check_var, check_var.name)
        types_dict = OrderedDict((a.var, set()) for a in funcs[0].arguments) #dict to collect each variable possible type and the corresponding flags
        # collect parse arg
        parse_args = [self.get_PyArgParseType(a.var) for a in funcs[0].arguments]

        # Managing the body of wrapper
        for func in funcs :
            mini_wrapper_func_body = []
            res_args = []
            static_func_args  = []

            mini_wrapper_func_name = self.scope.get_new_name(func.name + '_mini_wrapper')
            mini_scope = mod_scope.new_child_scope(mini_wrapper_func_name)
            self.set_scope(mini_scope)

            # update ndarray local variables properties
            arg_vars = {a.var: a for a in func.arguments}
            local_arg_vars = {(v.clone(v.name, memory_handling='alias')
                              if isinstance(v, Variable) and v.rank > 0 or v.is_optional \
                              else v) : a for v,a in arg_vars.items()}
            for a in local_arg_vars:
                mini_scope.insert_variable(a)
            for r in func.results:
                mini_scope.insert_variable(r)
            flags = 0

            # Loop for all args in every functions and create the corresponding condition and body
            for p_arg, (f_var, f_arg) in zip(parse_args, local_arg_vars.items()):
                collect_var  = self.get_PyArgParseType(f_var)
                body, tmp_variable = self._body_management(f_var, p_arg, f_arg.value)

                # get check type function
                check = self._get_check_type_statement(f_var, p_arg, f_arg.value is None)

                # Save the body
                wrapper_body_translations.extend(body)

                # Write default values
                if f_arg.value is not None:
                    wrapper_body.append(self.get_default_assign(parse_args[-1], f_var, f_arg.value))

                # Get Bind/C arguments
                static_func_args.extend(self.get_static_args(f_var))

                flag_value = flags_registry[(f_var.dtype, f_var.precision)]
                flags = (flags << 4) + flag_value  # shift by 4 to the left
                types_dict[f_var].add((f_var, check, flag_value)) # collect variable type for each arguments
                mini_wrapper_func_body += body

            # create the corresponding function call
            mini_wrapper_func_body.extend(self._get_static_func_call_code(func, static_func_args, func.results))


            # Loop for all res in every functions and create the corresponding body and cast
            for r in func.results :
                collect_var, cast_func = self.get_PyBuildValue(r)
                if cast_func is not None:
                    mini_wrapper_func_body.append(AliasAssign(collect_var, cast_func))

                res_args.append(ObjectAddress(collect_var) if collect_var.is_alias else collect_var)

            # Building PybuildValue and freeing the allocated variable after.
            mini_wrapper_func_body.append(AliasAssign(wrapper_results[0],PyBuildValueNode(res_args)))
            mini_wrapper_func_body += [FunctionCall(Py_DECREF, [i]) for i in self._to_free_PyObject_list]

            # Call free function for C type
            mini_wrapper_func_body += [If(IfSection(PyccelIsNot(i, Nil()), [Deallocate(i)])) if self.is_c_pointer(i) \
                                        else Deallocate(i) for i in local_arg_vars if i.rank > 0]
            mini_wrapper_func_body.append(Return(wrapper_results))
            self._to_free_PyObject_list.clear()

            self.set_scope(scope)

            # Building Mini wrapper function
            mini_wrapper_func_def = FunctionDef(name = mini_wrapper_func_name,
                arguments = parse_args,
                results = wrapper_results,
                body = mini_wrapper_func_body,
                scope = mini_scope)
            funcs_def.append(mini_wrapper_func_def)

            # append check condition to the functioncall
            body_tmp.append(IfSection(PyccelEq(check_var, LiteralInteger(flags)), [AliasAssign(wrapper_results[0],
                    FunctionCall(mini_wrapper_func_def, parse_args))]))

        # Errors / Types management
        # Creating check_type function
        check_func_def = self._create_wrapper_check(check_var, parse_args, types_dict)
        funcs_def.append(check_func_def)

        # Create the wrapper body with collected informations
        body_tmp = [IfSection(PyccelNot(check_var), [Return([Nil()])])] + body_tmp
        body_tmp.append(IfSection(LiteralTrue(),
            [PyErr_SetString('PyExc_TypeError', '"This combination of arguments is not valid"'),
            Return([Nil()])]))
        wrapper_body_translations = [If(*body_tmp)]

        # Parsing Arguments
        parse_node = PyArg_ParseTupleNode(*wrapper_args[1:],
                                          funcs[0].arguments,
                                          parse_args, keyword_list)

        wrapper_body += list(default_value.values())
        wrapper_body.append(If(IfSection(PyccelNot(parse_node), [Return([Nil()])])))

        #finishing the wrapper body
        wrapper_body.append(Assign(check_var, FunctionCall(check_func_def, parse_args)))
        wrapper_body.extend(wrapper_body_translations)
        wrapper_body.append(Return(wrapper_results)) # Return

        # Create FunctionDef
        funcs_def.append(FunctionDef(name = wrapper_name,
            arguments = wrapper_args,
            results = wrapper_results,
            body = wrapper_body,
            scope = scope))

        sep = self._print(SeparatorComment(40))

        self.exit_scope()

        return sep + '\n'.join(CCodePrinter._print_FunctionDef(self, f) for f in funcs_def)

    def _create_wrapper_check(self, check_var, parse_args, types_dict):
        """
        Create a FunctionDef which checks if the type is correct.

        This function is necessary when wrapping an Interface in order to determine which
        underlying function is being called. The created FunctionDef returns an integer
        which acts as a flag to indicate which function has been chosed.

        Parameters
        ----------
        check_var : Variable
            The variable which will contain the flag.
        parse_args : list of Variables
            The arguments to the Interface.
        types_dict : dictionary
            A dictionary listing all possible datatypes for each argument.

        Returns
        -------
        FunctionDef
            A FunctionDef describing the function which allows the flag to be set.
        """
        check_func_body = []
        flags = (len(types_dict) - 1) * 4
        for arg in types_dict:
            var_name = ""
            body = []
            types = []
            arg_type_check_list = list(types_dict[arg])
            arg_type_check_list.sort(key= lambda x : x[0].precision)
            for elem in arg_type_check_list:
                var_name = elem[0].name
                value = elem[2] << flags
                body.append(IfSection(elem[1], [AugAssign(check_var, '+' ,value)]))
                types.append(elem[0])
            flags -= 4
            description = [PythonType(v).print_string for v in types]
            error = ' or '.join([d.python_value for d in description])
            body.append(IfSection(LiteralTrue(), [PyErr_SetString('PyExc_TypeError', '"{} must be {}"'.format(var_name, error)), Return([LiteralInteger(0)])]))
            check_func_body += [If(*body)]

        check_func_body = [Assign(check_var, LiteralInteger(0))] + check_func_body
        check_func_body.append(Return([check_var]))
        # Creating check function definition
        check_func_name = self.scope.parent_scope.get_new_name('type_check')
        check_func_def = FunctionDef(name = check_func_name,
            arguments = parse_args,
            results = [check_var],
            body = check_func_body,
            scope = self.scope.new_child_scope(check_func_name))
        return check_func_def

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

    def _print_FunctionDef(self, expr):
        self.set_scope(self.scope.new_child_scope(expr.name))

        local_arg_vars = {}
        for a in expr.arguments:
            v = a.var
            if isinstance(v, Variable):
                new_name = self.scope.get_new_name(v.name)
                if isinstance(v, Variable) and (v.rank > 0 or v.is_optional):
                    new_v = v.clone(new_name, memory_handling='alias')
                else:
                    new_v = v.clone(new_name)
                local_arg_vars[new_v] = a
                self.scope.insert_variable(new_v)
            else:
                self.scope.functions[v.name] = v
                local_arg_vars[v] = a

        result_vars = [v.clone(self.scope.get_new_name(v.name)) for v in expr.results]
        for v in result_vars:
            self.scope.insert_variable(v)
        # update ndarray and optional local variables properties

        # Find a name for the wrapper function
        wrapper_name = self._get_wrapper_name(expr)

        # Collect arguments and results
        wrapper_args    = self.get_wrapper_arguments()
        wrapper_results = [self.get_new_PyObject("result")]
        self.scope.insert_variable(wrapper_results[0])

        # Collect argument names for PyArgParse
        arg_names         = [a.var.name for a in local_arg_vars.values()]
        keyword_list_name = self.scope.get_new_name('kwlist')
        keyword_list      = PyArgKeywords(keyword_list_name, arg_names)

        if expr.is_private:
            self.exit_scope()
            return self.untranslatable_function(wrapper_name,
                        wrapper_args, wrapper_results,
                        "Private functions are not accessible from python")

        if any(isinstance(arg, FunctionAddress) for arg in local_arg_vars):
            self.exit_scope()
            return self.untranslatable_function(wrapper_name,
                        wrapper_args, wrapper_results,
                        "Cannot pass a function as an argument")

        wrapper_body              = [keyword_list]
        wrapper_body_translations = []
        static_func_args  = []

        parse_args = []
        for var, arg in local_arg_vars.items():
            collect_var  = self.get_PyArgParseType(var)

            body, tmp_variable = self._body_management(var, collect_var, arg.value, True)

            # Save cast to argument variable
            wrapper_body_translations.extend(body)

            parse_args.append(collect_var)

            # Get Bind/C arguments
            static_func_args.extend(self.get_static_args(var))

            # Write default values
            if arg.value is not None:
                wrapper_body.append(self.get_default_assign(parse_args[-1], var, arg.value))

        # Parse arguments
        parse_node = PyArg_ParseTupleNode(*wrapper_args[1:],
                                          list(local_arg_vars.values()),
                                          parse_args, keyword_list)

        wrapper_body.append(If(IfSection(PyccelNot(parse_node), [Return([Nil()])])))
        wrapper_body.extend(wrapper_body_translations)

        wrapper_body.extend(self._get_static_func_call_code(expr, static_func_args, result_vars))

        # Loop over results to carry out necessary casts and collect Py_BuildValue type string
        res_args = []
        for a in result_vars :
            collect_var, cast_func = self.get_PyBuildValue(a)
            if cast_func is not None:
                wrapper_body.append(AliasAssign(collect_var, cast_func))

            res_args.append(ObjectAddress(collect_var) if collect_var.is_alias else collect_var)

        # Call PyBuildNode
        wrapper_body.append(AliasAssign(wrapper_results[0],PyBuildValueNode(res_args)))

        # Call free function for python type
        wrapper_body += [FunctionCall(Py_DECREF, [i]) for i in self._to_free_PyObject_list]

        # Call free function for C type
        wrapper_body += [If(IfSection(PyccelIsNot(i, Nil()), [Deallocate(i)])) if self.is_c_pointer(i) \
                            else Deallocate(i) for i in local_arg_vars if i.rank > 0]
        self._to_free_PyObject_list.clear()

        #Return
        wrapper_body.append(Return(wrapper_results))

        # Create FunctionDef and write using classic method
        wrapper_func = FunctionDef(name = wrapper_name,
            arguments = wrapper_args,
            results = wrapper_results,
            body = wrapper_body,
            scope = self.scope)

        self.exit_scope()

        return CCodePrinter._print_FunctionDef(self, wrapper_func)

    def _print_Module(self, expr):
        scope = Scope()
        self.set_scope(scope)
        # The initialisation and deallocation shouldn't be exposed to python
        funcs_to_wrap = [f for f in expr.funcs if f not in (expr.init_func, expr.free_func)]

        # Insert declared objects into scope
        if self._target_language == 'fortran':
            for f in expr.funcs:
                scope.insert_symbol('bind_c_'+f.name.lower())
            for v in expr.variables:
                if not v.is_private:
                    if v.rank > 0:
                        scope.insert_symbol('bind_c_'+v.name.lower())
                    else:
                        scope.insert_symbol(v.name.lower())
        else:
            for f in expr.funcs:
                scope.insert_symbol(f.name.lower())
            for v in expr.variables:
                if not v.is_private:
                    scope.insert_symbol(v.name.lower())

        if self._target_language == 'fortran':
            vars_to_wrap_decs = [Declare(v.dtype, v.clone(v.name.lower()), module_variable=True) \
                                    for v in expr.variables if not v.is_private and v.rank == 0]
        else:
            vars_to_wrap_decs = [Declare(v.dtype, v, module_variable=True) \
                                    for v in expr.variables if not v.is_private]

        self._module_name  = expr.name
        sep = self._print(SeparatorComment(40))

        if self._target_language == 'fortran':
            static_funcs = [self.get_static_function(f) for f in expr.funcs]
        else:
            static_funcs = expr.funcs
        function_signatures = ''.join('{};\n'.format(self.static_function_signature(f)) for f in static_funcs)
        if self._target_language == 'fortran':
            var_wrappers = [wrap_module_array_var(v, self.scope, expr) \
                    for v in expr.variables if not v.is_private and v.rank > 0]
            function_signatures += ''.join('{};\n'.format(self.function_signature(v)) for v in var_wrappers)

        interface_funcs = [f.name for i in expr.interfaces for f in i.functions]
        funcs = [*expr.interfaces, *(f for f in funcs_to_wrap if f.name not in interface_funcs)]

        self._in_header = True
        decs = ''.join('extern '+self._print(d) for d in vars_to_wrap_decs)
        self._in_header = False

        function_defs = '\n'.join(self._print(f) for f in funcs)
        cast_functions = '\n'.join(CCodePrinter._print_FunctionDef(self, f)
                                       for f in self._cast_functions_dict.values())
        method_def_func = ''.join(('{{\n'
                                     '"{name}",\n'
                                     '(PyCFunction){wrapper_name},\n'
                                     'METH_VARARGS | METH_KEYWORDS,\n'
                                     '{doc_string}\n'
                                     '}},\n').format(
                                            name = expr.scope.get_python_name(f.name),
                                            wrapper_name = self._function_wrapper_names[f.name],
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
                            method_def_func = method_def_func))

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
                    mod_name = expr.name,
                    method_def_name = method_def_name,
                    slots_name = slots_name))

        exec_func = self.get_module_exec_function(expr, exec_func_name)

        init_func = ('PyMODINIT_FUNC PyInit_{mod_name}(void)\n{{\n'
                'import_array();\n'
                'return PyModuleDef_Init(&{module_def_name});\n'
                '}}\n'.format(mod_name=expr.name,
                    module_def_name = module_def_name))

        # Print imports last to be sure that all additional_imports have been collected
        imports  = module_imports.copy()
        imports += self._additional_imports.values()
        imports  = ''.join(self._print(i) for i in imports)

        self.exit_scope()

        return ('#define PY_ARRAY_UNIQUE_SYMBOL CWRAPPER_ARRAY_API\n'
                '{imports}\n'
                '{variable_declarations}\n'
                '{function_signatures}\n'
                '{sep}\n'
                '{cast_functions}\n'
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
                    variable_declarations = decs,
                    function_signatures = function_signatures,
                    sep = sep,
                    cast_functions = cast_functions,
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
