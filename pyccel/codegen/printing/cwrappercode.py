# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=R0201

from collections import OrderedDict

from pyccel.codegen.printing.ccode import CCodePrinter

from pyccel.ast.bind_c   import as_static_function

from pyccel.ast.core import Assign, AliasAssign, FunctionDef, FunctionAddress
from pyccel.ast.core import If, IfSection, Return, FunctionCall, Deallocate
from pyccel.ast.core import create_incremented_string, SeparatorComment
from pyccel.ast.core import Import, Module, Declare
from pyccel.ast.core import AugAssign, CodeBlock

from pyccel.ast.cwrapper    import PyArg_ParseTupleNode, PyBuildValueNode
from pyccel.ast.cwrapper    import PyArgKeywords
from pyccel.ast.cwrapper    import Py_None, Py_DECREF
from pyccel.ast.cwrapper    import generate_datatype_error, PyErr_SetString
from pyccel.ast.cwrapper    import scalar_object_check, flags_registry
from pyccel.ast.cwrapper    import PyccelPyArrayObject, PyccelPyObject
from pyccel.ast.cwrapper    import C_to_Python, Python_to_C
from pyccel.ast.cwrapper    import PyModule_AddObject

from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeFloat, str_dtype
from pyccel.ast.datatypes import datatype

from pyccel.ast.literals  import LiteralTrue, LiteralInteger, LiteralString
from pyccel.ast.literals  import Nil

from pyccel.ast.numpy_wrapper   import array_checker, array_type_check
from pyccel.ast.numpy_wrapper   import pyarray_to_ndarray
from pyccel.ast.numpy_wrapper   import array_get_data, array_get_dim

from pyccel.ast.operators import PyccelEq, PyccelNot, PyccelOr, PyccelAssociativeParenthesis
from pyccel.ast.operators import PyccelIsNot, PyccelLt, PyccelUnarySub

from pyccel.ast.variable  import VariableAddress, Variable

from pyccel.errors.errors   import Errors

errors = Errors()

__all__ = ["CWrapperCodePrinter", "cwrappercode"]

dtype_registry = {('pyobject'     , 0) : 'PyObject',
                  ('pyarrayobject', 0) : 'PyArrayObject'}

module_imports  = [Import('numpy_version', Module('numpy_version',(),())),
            Import('numpy/arrayobject', Module('numpy/arrayobject',(),())),
            Import('cwrapper', Module('cwrapper',(),()))]

cwrapper_ndarray_import = Import('cwrapper_ndarrays', Module('cwrapper_ndarrays', (), ()))

class CWrapperCodePrinter(CCodePrinter):
    """A printer to convert a python module to strings of c code creating
    an interface between python and an implementation of the module in c"""
    def __init__(self, parser, target_language, **settings):
        CCodePrinter.__init__(self, parser, **settings)
        self._target_language = target_language
        self._cast_functions_dict = OrderedDict()
        self._to_free_PyObject_list = []
        self._function_wrapper_names = dict()
        self._global_names = set()
        self._module_name = None


    # --------------------------------------------------------------------
    #                       Helper functions
    # --------------------------------------------------------------------

    def stored_in_c_pointer(self, a):
        """
        Indicates whether the object a needs to be stored in a pointer
        in c code

        Parameters
        ----------
        a : PyccelAstNode
        """
        if isinstance(a.dtype, (PyccelPyArrayObject, PyccelPyObject)):
            return True
        else:
            return CCodePrinter.stored_in_c_pointer(self,a)

    def get_new_name(self, used_names, requested_name):
        """
        Generate a new name, return the requested_name if it's not in
        used_names set  or generate new one based on the requested_name.
        The generated name is appended to the used_names set

        Parameters
        ----------
        used_names     : set of strings
            Set of variable and function names to avoid name collisions
        requested_name : String
            The desired name

        Returns
        ----------
        name  : String
        """
        if requested_name not in used_names:
            used_names.add(requested_name)
            return requested_name
        else:
            incremented_name, _ = create_incremented_string(used_names, prefix=requested_name)
            return incremented_name

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
        Get the declaration type of a variable

        Parameters
        -----------
        variable : Variable
            Variable holding information needed to choose the declaration type

        Returns
        -------
        type_declaration : String
        """
        dtype = self._print(expr.dtype)
        prec  = expr.precision
        if dtype != "pyarrayobject":
            #TODO: Remove when #757 is fixed
            if expr.rank > 0 and expr.is_ndarray and expr.is_optional:
                dtype = 't_ndarray'
            else:
                return CCodePrinter.get_declare_type(self, expr)
        else :
            dtype = self.find_in_dtype_registry(dtype, prec)

        if self.stored_in_c_pointer(expr):
            return '{0} *'.format(dtype)
        else:
            return '{0} '.format(dtype)

    def get_new_PyObject(self, name, used_names):
        """
        Create new PyccelPyObject Variable with the desired name

        Parameters
        -----------
        name       : String
            The desired name
        used_names : Set of strings
            Set of variable and function names to avoid name collisions

        Returns: Variable
        -------
        """
        return Variable(dtype=PyccelPyObject(),
                        name=self.get_new_name(used_names, name),
                        is_pointer=True)

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
        if arg.rank > 0 :
            return AliasAssign(arg, Nil())
        elif func_arg.is_optional:
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
            static_func = as_static_function(function)
        else:
            static_func = function

        return static_func

    def static_function_signature(self, expr):
        """
        Extract from the function definition all the information (name, input, output)
        needed to create the function signature used for C/fortran binding

        Parameters:
        ----------
        expr : FunctionDef
            The function defintion

        Return:
        ------
        String
            Signature of the function
        """
        #if target_language is C no need for the binding
        if self._target_language == 'c':
            return self.function_signature(expr)

        args = [a.var for a in expr.arguments]
        if len(expr.results) == 1:
            ret_type = self.get_declare_type(expr.results[0])
        elif len(expr.results) > 1:
            ret_type = self._print(datatype('int')) + ' '
            args += [a.clone(name = a.name, is_pointer =True) for a in expr.results]
        else:
            ret_type = self._print(datatype('void')) + ' '
        name = expr.name
        if not args:
            arg_code = 'void'
        else:
            arg_code = ', '.join('{}'.format(self.function_signature(i, False))
                        if isinstance(i, FunctionAddress)
                        else '{0}{1}'.format(self.get_static_declare_type(i), i.name)
                        for i in args)

        if isinstance(expr, FunctionAddress):
            return '{}(*{})({})'.format(ret_type, name, arg_code)
        else:
            return '{0}{1}({2})'.format(ret_type, name, arg_code)

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
            arg_address = VariableAddress(argument)
            static_args = [
                FunctionCall(array_get_dim, [arg_address, i]) for i in range(argument.rank)
            ]
            static_args.append(FunctionCall(array_get_data, [arg_address]))
        else:
            static_args = [argument]

        return static_args

    def get_static_declare_type(self, variable):
        """
        Get the declaration type of a variable, this function is used for
        C/fortran binding using native C datatypes.

        Parameters
        ----------
        variable : Variable
            Variable holding information needed to choose the declaration type

        Returns
        -------
        string

        """
        dtype = self._print(variable.dtype)
        prec  = variable.precision

        dtype = self.find_in_dtype_registry(dtype, prec)

        if self.stored_in_c_pointer(variable):
            return '{0} *'.format(dtype)

        elif self._target_language == 'fortran' and variable.rank > 0:
            return '{0} *'.format(dtype)

        else:
            return '{0} '.format(dtype)

    def _get_check_type_statement(self, variable, collect_var, compulsory):
        """
        Get the code which checks if the variable collected from python
        has the expected type

        Parameters
        ----------
        variable    : Variable
                      The variable containing the PythonObject
        collect_var : Variable
                      The variable in which the result will be saved,
                      used to provide information about the expected type
        compulsory  : bool
                      Indicates whether the argument is a compulsory argument
                      to the function (if not then it must have a default or
                      be optional)

        Returns
        -------
        check : str
                A string containing the code which determines whether 'variable'
                contains an object which can be saved in 'collect_var'
        """

        if variable.rank > 0 :
            check = array_type_check(collect_var, variable)

        else :
            check = scalar_object_check(collect_var, variable)

        if not compulsory:
            default = PyccelNot(VariableAddress(collect_var)) \
                            if variable.rank > 0 else \
                      PyccelEq(VariableAddress(collect_var), VariableAddress(Py_None))
            check = PyccelAssociativeParenthesis(PyccelOr(default, check))

        return check

    def _get_wrapper_name(self, used_names, func):
        """
        create wrapper function name

        Parameters
        -----------
        used_names: list of strings
            List of variable and function names to avoid name collisions
        func      : FunctionDef or Interface

        Returns
        -------
        wrapper_name : string
        """
        name         = func.name
        wrapper_name = self.get_new_name(used_names.union(self._global_names), name+"_wrapper")

        self._function_wrapper_names[func.name] = wrapper_name
        self._global_names.add(wrapper_name)

        return wrapper_name

    def get_wrapper_arguments(self, used_names):
        """
        Create wrapper arguments

        Parameters
        ----------
        used_names : Set of strings
            Set of variable and function names to avoid name collisions

        Returns
        -------
        List of variables
        """
        python_func_args    = self.get_new_PyObject("args"  , used_names)
        python_func_kwargs  = self.get_new_PyObject("kwargs", used_names)
        python_func_selfarg = self.get_new_PyObject("self"  , used_names)

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

        valued_var_check  = PyccelEq(VariableAddress(collect_var), VariableAddress(Py_None))
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
        Responsible for collecting value and managing error and create the body
        of arguments with rank greater than 0 in format

        Parameters
        ----------
        Variable : Variable
            The optional variable
        collect_var : variable
            the pyobject type variable  holder of value
        check_type : Boolean
            True if the type is needed

        Returns
        -------
        body : list
            A list of statements
        """
        body = []
        #check optional :
        if variable.is_optional :
            check = PyccelNot(VariableAddress(collect_var))
            body += [IfSection(check, [Assign(VariableAddress(variable), Nil())])]

        check = array_checker(collect_var, variable, check_type, self._target_language)
        body += [IfSection(check, [Return([Nil()])])]

        collect_func = FunctionCall(pyarray_to_ndarray, [collect_var])
        body += [IfSection(LiteralTrue(), [Assign(variable,
                            collect_func)])]
        body = [If(*body)]

        return body

    def _body_management(self, used_names, variable, collect_var, default_value = None, check_type = False):
        """
        Responsible for calling functions that take care of body creation

        Parameters
        ----------
        used_names    : set of strings
                        Set of variable and function names to avoid name collisions
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
                                        name = self.get_new_name(used_names, variable.name+"_tmp"))

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
        wrapper_func = FunctionDef(
                name      = wrapper_name,
                arguments = wrapper_args,
                results   = wrapper_results,
                body      = [
                                PyErr_SetString('PyExc_NotImplementedError',
                                            '"{}"'.format(error_msg)),
                                AliasAssign(wrapper_results[0], Nil()),
                                Return(wrapper_results)
                            ])

        return CCodePrinter._print_FunctionDef(self, wrapper_func)

    # -------------------------------------------------------------------
    # Parsing arguments and building values Types functions
    # -------------------------------------------------------------------
    def get_PyArgParseType(self, used_names, variable):
        """
        Responsible for creating any necessary intermediate variables which are used
        to collect the result of PyArgParse, and collecting the required cast function

        Parameters
        ----------
        used_names : list of strings
            List of variable and function names to avoid name collisions

        variable : Variable
            The variable which will be passed to the translated function

        Returns
        -------
        collect_var : Variable
            The variable which will be used to collect the argument
        """

        if variable.rank > 0:
            collect_type = PyccelPyArrayObject()
            collect_var  = Variable(dtype= collect_type, is_pointer = True, rank = variable.rank,
                                   order= variable.order,
                                   name=self.get_new_name(used_names, variable.name+"_tmp"))

        else:
            collect_type = PyccelPyObject()
            collect_var  = Variable(dtype=collect_type, is_pointer=True,
                                   name = self.get_new_name(used_names, variable.name+"_tmp"))

        return collect_var

    def get_PyBuildValue(self, used_names, variable):
        """
        Responsible for collecting the variable required to build the result
        and the necessary cast function

        Parameters
        ----------
        used_names : list of strings
            List of variable and function names to avoid name collisions

        variable : Variable
            The variable returned by the translated function

        Returns
        -------
        collect_var : Variable
            The variable which will be provided to PyBuild

        cast_func_stmts : functionCall
            call to cast function responsible for the conversion of one data type into another
        """

        cast_function = FunctionCall(C_to_Python(variable), [VariableAddress(variable)])

        collect_type = PyccelPyObject()
        collect_var = Variable(dtype=collect_type, is_pointer=True,
            name = self.get_new_name(used_names, variable.name+"_tmp"))
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
            if var.fst:
                symbol = var
            else:
                symbol = var.get_user_nodes(Assign)[0]
            errors.report("Global arrays (defined at the module level) cannot currently be exposed to Python",
                    severity='warning', symbol=symbol)
            return []

        collect_value = Assign(VariableAddress(collect_var),
                                FunctionCall(C_to_Python(var), [VariableAddress(var)]))
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
        used_names = set([exec_func_name] + [v.name for v in expr.variables])
        mod_var_name = self.get_new_name(used_names, 'm')
        tmp_var_name = self.get_new_name(used_names, 'tmp')
        tmp_var = Variable(dtype = PyccelPyObject(),
                      name       = tmp_var_name,
                      is_pointer = True)

        orig_vars_to_wrap = [v for v in expr.variables if not v.is_private]
        if self._target_language == 'fortran':
            vars_to_wrap = [v.clone(v.name.lower()) for v in orig_vars_to_wrap]
            for v,w in zip(orig_vars_to_wrap,vars_to_wrap):
                assign = v.get_user_nodes(Assign)[0]
                # assign.fst should always exist, but is not always set when the
                # Assign is created in the codegen stage
                if assign.fst:
                    w.set_fst(assign.fst)
        else:
            vars_to_wrap = orig_vars_to_wrap
        var_names = [v.name for v in orig_vars_to_wrap]

        body = [l for n,v in zip(var_names,vars_to_wrap) for l in self.insert_constant(mod_var_name, n, v, tmp_var)]

        decs = self._print(Declare(tmp_var.dtype, tmp_var)) if body else ''

        if expr.init_func:
            static_function = self.get_static_function(expr.init_func)
            body.insert(0,FunctionCall(static_function,[],[]))

        body.append(Return([LiteralInteger(0)]))

        body_str = self._print(CodeBlock(body))

        return ('static int {name}(PyObject* {mod_var})\n'
                '{{\n'
                '{decs}'
                '{body}'
                '}}\n').format(name = exec_func_name,
                        mod_var = mod_var_name,
                        decs = decs,
                        body = body_str)

    #--------------------------------------------------------------------
    #                 _print_ClassName functions
    #--------------------------------------------------------------------

    def _print_Interface(self, expr):

        # Collecting all functions
        funcs = expr.functions
        # Save all used names
        used_names = set(n.name for n in funcs)

        # Find a name for the wrapper function
        wrapper_name = self._get_wrapper_name(used_names, expr)
        used_names.add(wrapper_name)

        # Collect arguments and results
        wrapper_args    = self.get_wrapper_arguments(used_names)
        wrapper_results = [self.get_new_PyObject("result", used_names)]

        # Collect parser arguments
        wrapper_vars = {}

        # Collect argument names for PyArgParse
        arg_names         = [a.name for a in funcs[0].arguments]
        keyword_list_name = self.get_new_name(used_names,'kwlist')
        keyword_list      = PyArgKeywords(keyword_list_name, arg_names)
        wrapper_body      = [keyword_list]

        wrapper_body_translations = []
        body_tmp = []

        # To store the mini function responsible for collecting value and calling interfaces functions and return the builded value
        funcs_def = []
        default_value = {} # dict to collect all initialisation needed in the wrapper
        check_var = Variable(dtype = NativeInteger(), name = self.get_new_name(used_names , "check"))
        wrapper_vars[check_var.name] = check_var
        types_dict = OrderedDict((a.var, set()) for a in funcs[0].arguments) #dict to collect each variable possible type and the corresponding flags
        # collect parse arg
        parse_args = [self.get_PyArgParseType(used_names,a.var) for a in funcs[0].arguments]

        # Managing the body of wrapper
        for func in funcs :
            mini_wrapper_func_body = []
            res_args = []
            static_func_args  = []
            # update ndarray local variables properties
            arg_vars = {a.var: a for a in func.arguments}
            local_arg_vars = {(v.clone(v.name, is_pointer=True, allocatable=False)
                              if isinstance(v, Variable) and v.rank > 0 or v.is_optional \
                              else v) : a for v,a in arg_vars.items()}
            mini_wrapper_func_vars = {a.name : a for a in local_arg_vars}
            flags = 0
            collect_vars = {}

            # Loop for all args in every functions and create the corresponding condition and body
            for p_arg, (f_var, f_arg) in zip(parse_args, local_arg_vars.items()):
                collect_var  = self.get_PyArgParseType(used_names, f_var)
                collect_vars[f_var] = collect_var
                body, tmp_variable = self._body_management(used_names, f_var, p_arg, f_arg.value)
                if tmp_variable :
                    mini_wrapper_func_vars[tmp_variable.name] = tmp_variable

                # get check type function
                check = self._get_check_type_statement(f_var, p_arg, f_arg.value is None)
                # If the variable cannot be collected from PyArgParse directly
                wrapper_vars[p_arg.name] = p_arg

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
            static_function = self.get_static_function(func)
            mini_wrapper_func_vars.update({arg.name : arg for arg in static_func_args if isinstance(arg, Variable)})

            if len(func.results)==0:
                func_call = FunctionCall(static_function, static_func_args)
            else:
                results   = func.results if len(func.results)>1 else func.results[0]
                func_call = Assign(results,FunctionCall(static_function, static_func_args))

            mini_wrapper_func_body.append(func_call)

            # Loop for all res in every functions and create the corresponding body and cast
            for r in func.results :
                collect_var, cast_func = self.get_PyBuildValue(used_names, r)
                mini_wrapper_func_vars[collect_var.name] = collect_var
                if cast_func is not None:
                    mini_wrapper_func_vars[r.name] = r
                    mini_wrapper_func_body.append(AliasAssign(collect_var, cast_func))
                res_args.append(VariableAddress(collect_var) if collect_var.is_pointer else collect_var)

            # Building PybuildValue and freeing the allocated variable after.
            mini_wrapper_func_body.append(AliasAssign(wrapper_results[0],PyBuildValueNode(res_args)))
            mini_wrapper_func_body += [FunctionCall(Py_DECREF, [i]) for i in self._to_free_PyObject_list]

            # Call free function for C type
            mini_wrapper_func_body += [If(IfSection(PyccelIsNot(i, Nil()), [Deallocate(i)])) if self.stored_in_c_pointer(i) \
                                        else Deallocate(i) for i in local_arg_vars if i.rank > 0]
            mini_wrapper_func_body.append(Return(wrapper_results))
            self._to_free_PyObject_list.clear()

            # Building Mini wrapper function
            mini_wrapper_func_name = self.get_new_name(used_names.union(self._global_names), func.name + '_mini_wrapper')
            self._global_names.add(mini_wrapper_func_name)

            mini_wrapper_func_def = FunctionDef(name = mini_wrapper_func_name,
                arguments = parse_args,
                results = wrapper_results,
                body = mini_wrapper_func_body,
                local_vars = mini_wrapper_func_vars.values())
            funcs_def.append(mini_wrapper_func_def)

            # append check condition to the functioncall
            body_tmp.append(IfSection(PyccelEq(check_var, LiteralInteger(flags)), [AliasAssign(wrapper_results[0],
                    FunctionCall(mini_wrapper_func_def, parse_args))]))

        # Errors / Types management
        # Creating check_type function
        check_func_def = self._create_wrapper_check(check_var, parse_args, types_dict, used_names, funcs[0].name)
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
            local_vars = wrapper_vars.values()))

        sep = self._print(SeparatorComment(40))

        return sep + '\n'.join(CCodePrinter._print_FunctionDef(self, f) for f in funcs_def)

    def _create_wrapper_check(self, check_var, parse_args, types_dict, used_names, func_name):
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
            error = ' or '.join(['{} {}'.format(str(v.precision * 8)+' bit' if v.precision != -1 else 'native',
                                                    str_dtype(v.dtype))
                            if not isinstance(v.dtype, NativeBool)
                            else  str_dtype(v.dtype) for v in types])
            body.append(IfSection(LiteralTrue(), [PyErr_SetString('PyExc_TypeError', '"{} must be {}"'.format(var_name, error)), Return([LiteralInteger(0)])]))
            check_func_body += [If(*body)]

        check_func_body = [Assign(check_var, LiteralInteger(0))] + check_func_body
        check_func_body.append(Return([check_var]))
        # Creating check function definition
        check_func_name = self.get_new_name(used_names.union(self._global_names), 'type_check')
        self._global_names.add(check_func_name)
        check_func_def = FunctionDef(name = check_func_name,
            arguments = parse_args,
            results = [check_var],
            body = check_func_body,
            local_vars = [])
        return check_func_def

    def _print_PyccelPyObject(self, expr):
        return 'pyobject'

    def _print_PyccelPyArrayObject(self, expr):
        self.add_import(cwrapper_ndarray_import)
        return 'pyarrayobject'

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
            code = '{name}("{flags}", {args})'.format(name=name, flags=flags, args=args)
        else :
            code = '{name}("")'.format(name=name)
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
        # Save all used names
        used_names = set([a.name for a in expr.arguments]
                        + [r.name for r in expr.results]
                        + [expr.name])

        arg_vars = {a.var: a for a in expr.arguments}
        # update ndarray and optional local variables properties
        local_arg_vars = {(v.clone(v.name, is_pointer=True, allocatable=False)
                          if isinstance(v, Variable) and (v.rank > 0 or v.is_optional) \
                          else v) : a for v,a in arg_vars.items()}

        # Find a name for the wrapper function
        wrapper_name = self._get_wrapper_name(used_names, expr)
        used_names.add(wrapper_name)
        # Collect local variables
        wrapper_vars        = {a.name : a for a in local_arg_vars}
        wrapper_vars.update({r.name : r for r in expr.results})

        # Collect arguments and results
        wrapper_args    = self.get_wrapper_arguments(used_names)
        wrapper_results = [self.get_new_PyObject("result", used_names)]

        # Collect argument names for PyArgParse
        arg_names         = [a.name for a in local_arg_vars]
        keyword_list_name = self.get_new_name(used_names,'kwlist')
        keyword_list      = PyArgKeywords(keyword_list_name, arg_names)

        if expr.is_private:
            return self.untranslatable_function(wrapper_name,
                        wrapper_args, wrapper_results,
                        "Private functions are not accessible from python")

        if any(isinstance(arg, FunctionAddress) for arg in local_arg_vars):
            return self.untranslatable_function(wrapper_name,
                        wrapper_args, wrapper_results,
                        "Cannot pass a function as an argument")

        wrapper_body              = [keyword_list]
        wrapper_body_translations = []
        static_func_args  = []

        parse_args = []
        collect_vars = {}
        for var, arg in local_arg_vars.items():
            collect_var  = self.get_PyArgParseType(used_names, var)
            collect_vars[var] = collect_var

            body, tmp_variable = self._body_management(used_names, var, collect_var, arg.value, True)
            if tmp_variable :
                wrapper_vars[tmp_variable.name] = tmp_variable

            # If the variable cannot be collected from PyArgParse directly
            wrapper_vars[collect_var.name] = collect_var

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

        # Call function
        static_function = self.get_static_function(expr)
        wrapper_vars.update({arg.name : arg for arg in static_func_args if isinstance(arg, Variable)})

        if len(expr.results)==0:
            func_call = FunctionCall(static_function, static_func_args)
        else:
            results   = expr.results if len(expr.results)>1 else expr.results[0]
            func_call = Assign(results,FunctionCall(static_function, static_func_args))

        wrapper_body.append(func_call)

        # Loop over results to carry out necessary casts and collect Py_BuildValue type string
        res_args = []
        for a in expr.results :
            collect_var, cast_func = self.get_PyBuildValue(used_names, a)
            if cast_func is not None:
                wrapper_vars[collect_var.name] = collect_var
                wrapper_body.append(AliasAssign(collect_var, cast_func))

            res_args.append(VariableAddress(collect_var) if collect_var.is_pointer else collect_var)

        # Call PyBuildNode
        wrapper_body.append(AliasAssign(wrapper_results[0],PyBuildValueNode(res_args)))

        # Call free function for python type
        wrapper_body += [FunctionCall(Py_DECREF, [i]) for i in self._to_free_PyObject_list]

        # Call free function for C type
        wrapper_body += [If(IfSection(PyccelIsNot(i, Nil()), [Deallocate(i)])) if self.stored_in_c_pointer(i) \
                            else Deallocate(i) for i in local_arg_vars if i.rank > 0]
        self._to_free_PyObject_list.clear()

        #Return
        wrapper_body.append(Return(wrapper_results))

        # Create FunctionDef and write using classic method
        wrapper_func = FunctionDef(name = wrapper_name,
            arguments = wrapper_args,
            results = wrapper_results,
            body = wrapper_body,
            local_vars = tuple(wrapper_vars.values()))

        return CCodePrinter._print_FunctionDef(self, wrapper_func)

    def _print_Module(self, expr):
        # The initialisation and deallocation shouldn't be exposed to python
        funcs_to_wrap = [f for f in expr.funcs if f not in (expr.init_func, expr.free_func)]

        if self._target_language == 'fortran':
            vars_to_wrap_decs = [Declare(v.dtype, v.clone(v.name.lower()), module_variable=True) \
                                    for v in expr.variables if not v.is_private and v.rank == 0]
        else:
            vars_to_wrap_decs = [Declare(v.dtype, v, module_variable=True) \
                                    for v in expr.variables if not v.is_private and v.rank == 0]

        self._global_names = set(f.name for f in expr.funcs)
        self._module_name  = expr.name
        sep = self._print(SeparatorComment(40))

        if self._target_language == 'fortran':
            static_funcs = [self.get_static_function(f) for f in expr.funcs]
        else:
            static_funcs = expr.funcs
        function_signatures = ''.join('{};\n'.format(self.static_function_signature(f)) for f in static_funcs)

        interface_funcs = [f.name for i in expr.interfaces for f in i.functions]
        funcs = [*expr.interfaces, *(f for f in funcs_to_wrap if f.name not in interface_funcs)]

        decs = ''.join('extern '+self._print(d) for d in vars_to_wrap_decs)

        function_defs = '\n'.join(self._print(f) for f in funcs)
        cast_functions = '\n'.join(CCodePrinter._print_FunctionDef(self, f)
                                       for f in self._cast_functions_dict.values())
        method_def_func = ''.join(('{{\n'
                                     '"{name}",\n'
                                     '(PyCFunction){wrapper_name},\n'
                                     'METH_VARARGS | METH_KEYWORDS,\n'
                                     '{doc_string}\n'
                                     '}},\n').format(
                                            name = f.name,
                                            wrapper_name = self._function_wrapper_names[f.name],
                                            doc_string = self._print(LiteralString('\n'.join(f.doc_string.comments))) \
                                                        if f.doc_string else '""')
                                     for f in funcs if f is not expr.init_func)

        slots_name = self.get_new_name(self._global_names, '{}_slots'.format(expr.name))
        exec_func_name = self.get_new_name(self._global_names, 'exec_func')
        slots_def = ('static PyModuleDef_Slot {name}[] = {{\n'
                     '{{Py_mod_exec, {exec_func}}},\n'
                     '{{0, NULL}},\n'
                     '}};\n').format(name = slots_name,
                             exec_func = exec_func_name)

        method_def_name = self.get_new_name(self._global_names, '{}_methods'.format(expr.name))
        method_def = ('static PyMethodDef {method_def_name}[] = {{\n'
                        '{method_def_func}'
                        '{{ NULL, NULL, 0, NULL}}\n'
                        '}};\n'.format(method_def_name = method_def_name ,method_def_func = method_def_func))

        module_def_name = self.get_new_name(self._global_names, '{}_module'.format(expr.name))
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

def cwrappercode(expr, parser, target_language, assign_to=None, **settings):
    """Converts an expr to a string of c wrapper code

    expr : Expr
        A pyccel expression to be converted.
    parser : Parser
        The parser used to collect the expression
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

    return CWrapperCodePrinter(parser, target_language, **settings).doprint(expr, assign_to)
