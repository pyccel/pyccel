# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=R0201

from collections import OrderedDict

from pyccel.codegen.printing.ccode import CCodePrinter

from pyccel.ast.literals    import LiteralTrue, LiteralInteger, LiteralString

from pyccel.ast.operators   import PyccelNot, PyccelEq, PyccelIs, PyccelIsNot, PyccelNe

from pyccel.ast.datatypes   import NativeInteger, NativeGeneric, NativeBool, NativeComplex

from pyccel.ast.core        import create_incremented_string, SeparatorComment

from pyccel.ast.core        import FunctionCall, FunctionDef, FunctionAddress
from pyccel.ast.core        import Assign, AliasAssign, Nil, datatype
from pyccel.ast.core        import If, IfSection, Import, Return, Deallocate

from pyccel.ast.cwrapper    import PyArgKeywords, PyArg_ParseTupleNode, PyBuildValueNode
from pyccel.ast.cwrapper    import Py_CLEANUP_SUPPORTED

from pyccel.ast.cwrapper    import C_to_Python, Python_to_C, get_custom_key

from pyccel.ast.cwrapper    import flags_registry, PyErr_SetString, Py_None, PyErr_Occurred
from pyccel.ast.cwrapper    import malloc, free, sizeof, generate_datatype_error

from pyccel.ast.cwrapper    import PyccelPyObject, PyccelPyArrayObject, scalar_object_check

from pyccel.ast.numpy_wrapper   import array_checker, array_get_dim, array_get_data
from pyccel.ast.numpy_wrapper   import pyarray_to_f_ndarray, pyarray_to_c_ndarray

from pyccel.ast.variable        import Variable, ValuedVariable, VariableAddress

from pyccel.ast.bind_c          import as_static_function_call

from pyccel.errors.errors   import Errors

errors = Errors()

__all__ = ["CWrapperCodePrinter", "cwrappercode"]

dtype_registry = {('pyobject'     , 0) : 'PyObject',
                  ('pyarrayobject', 0) : 'PyArrayObject'}

RETURN_ZERO = Return([LiteralInteger(0)])
RETURN_NULL = Return([Nil()])

class CWrapperCodePrinter(CCodePrinter):
    """A printer to convert a python module to strings of c code creating
    an interface between python and an implementation of the module in c"""
    def __init__(self, parser, target_language, **settings):
        CCodePrinter.__init__(self, parser, **settings)
        self._target_language             = target_language
        self._function_wrapper_names      = dict()
        self._global_names                = set()
        self._module_name                 = None
        self._converter_functions         = dict()
        self.to_free_objects              = []
    # --------------------------------------------------------------------
    #                       Helper functions
    # --------------------------------------------------------------------
    @staticmethod
    def get_new_name(used_names, requested_name):
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
            name = requested_name
        else:
            name, _ = create_incremented_string(used_names, prefix=requested_name)

        return name

    def get_wrapper_name(self, used_names, function):
        """
        Generate wrapper function name
        Parameters:
        -----------
        used_names : set of strings
            Set of variable and function names to avoid name collisions

        function   : FunctionDef or Interface

        Returns:
        -------
        wrapper_name : string
        """
        name = function.name
        wrapper_name = self.get_new_name(used_names.union(self._global_names), name+"_wrapper")

        self._function_wrapper_names[name] = wrapper_name
        self._global_names.add(wrapper_name)
        used_names.add(wrapper_name)

        return wrapper_name

    def get_new_PyObject(self, name, used_names):
        """
        Create new PyccelPyObject Variable with the desired name

        Parameters:
        -----------
        name       : String
            The desired name

        used_names : Set of strings
            Set of variable and function names to avoid name collisions
    
        Returns: Variable
        -------
        """
        dtype = PyccelPyObject()

        return Variable(dtype      = dtype,
                        name       = self.get_new_name(used_names, name),
                        is_pointer = True)


    def get_wrapper_arguments(self, used_names):
        """
        Create wrapper arguments
        Parameters:
        -----------
        used_names : Set of strings
            Set of variable and function names to avoid name collisions

        Returns: List of variables
        """
        python_func_args    = self.get_new_PyObject("args"  , used_names)
        python_func_kwargs  = self.get_new_PyObject("kwargs", used_names)
        python_func_selfarg = self.get_new_PyObject("self"  , used_names)

        return [python_func_selfarg, python_func_args, python_func_kwargs]

    def find_in_dtype_registry(self, dtype, prec):
        """
        find the corresponding C dtype in the dtype_registry
        raise PYCCEL_RESTRICTION_TODO if not found

        Parameters:
        -----------
        dtype : String
            expression data type

        prec  : Integer
            expression precision

        Returns: String
        --------
        """
        try :
            return dtype_registry[(dtype, prec)]
        except KeyError:
            return CCodePrinter.find_in_dtype_registry(self, dtype, prec)

    def get_declare_type(self, variable):
        """
        Get the declaration type of a variable

        Parameters:
        -----------
        variable : Variable
            Variable holding information needed to choose the declaration type

        Returns: String
        --------

        """
        dtype = self._print(variable.dtype)
        prec  = variable.precision
        rank  = variable.rank

        dtype = self.find_in_dtype_registry(dtype, prec)
        if rank > 0:
            if variable.is_ndarray:
                dtype = 't_ndarray'
            else:
                errors.report(PYCCEL_RESTRICTION_TODO, symbol="rank > 0",severity='fatal')

        if variable.is_pointer and variable.is_optional:
            return '{} **'.format(dtype)
    
        if self.stored_in_c_pointer(variable):
            return '{0} *'.format(dtype)

        return '{0} '.format(dtype)

    def get_static_declare_type(self, variable):
        """
        Get the declaration type of a variable, this function is used for
        C/fortran binding using native C datatypes.

        Parameters:
        -----------
        variable : Variable
            Variable holding information needed to choose the declaration type

        Returns: String
        --------

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

        args = list(expr.arguments)
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

    def stored_in_c_pointer(self, expr):
        """
        Return True if variable is pointer or stored in pointer

        Parameters:
        -----------
        a      : Variable
            Variable holding information needed (is_pointer, is_optional)

        Returns: boolean
        --------
        """
        if not isinstance(expr, Variable):
            return False

        return expr.is_pointer or expr.is_optional

    def get_static_function(self, function):
        """
        Create a static FunctionDef from the argument used for
        C/fortran binding.
        If target language is C return the argument function

        Parameters:
        -----------
        function    : FunctionDef
            FunctionDef holding information needed to create static function

        Returns   :
        -----------
        static_func : FunctionDef
        """
        if self._target_language == 'fortran':
            static_func = as_static_function_call(function,
                                                  self._module_name,
                                                  name = function.name)
        else:
            static_func = function

        return static_func

    def get_static_args(self, argument):
        """
        Create bind_C arguments for arguments rank > 0 in fortran.
        needed in static function call
        func(a) ==> static_func(nd_dim(a) , nd_data(a))
        where nd_data(a) = buffer holding data
              nd_dim(a)  = size of array

        Parameters:
        -----------
        argument    : Variable
            Variable holding information needed (rank)

        Returns     : List of arguments
            List that can contains Variables and FunctionCalls
        -----------
        """

        if self._target_language == 'fortran' and argument.rank > 0:
            static_args = [
                FunctionCall(array_get_dim, [argument, i]) for i in range(argument.rank)
            ]

            static_args.append(FunctionCall(array_get_data, [argument]))
        else:
            static_args = [argument]

        return static_args

    @staticmethod
    def set_flag_value(flag, variable):
        """
        Collect data type flag value from flags_registry used to avoid
        multiple data type check when using interfaces, and set the 
        new flag value, raise NotImplementedError if not found

        Parameters:
        -----------
        flag     : Integer
            the current flag value

        variable : Variable
            Variable holding information needed (dtype, precision)

        Returns  : Integer
        -------
            the new flag value
        """
        try :
            new_flag = flags_registry[(variable.dtype, variable.precision)]
        except KeyError:
            raise NotImplementedError(
            'datatype not implemented as arguments : {}'.format(variable.dtype))
        return (flag << 4) + flag


    def get_default_assign(self, variable):
        """
        Look up for the default value and create default assign
        
        Parameters:
        -----------
        variable : Variable
            Variable holding information needed (value)

        Returns  : Assign
        -------
        """

        if variable.is_optional:
            assign = Assign(VariableAddress(variable), Nil())

        else: #valued
            assign = Assign(variable, variable.value)

        return assign

    def get_free_statements(self, variable):
        """
        """
        body = []
        # once valued variable rank > 0 are implemented change should be made here
        if variable.rank > 0 and self._target_language is 'c':
            if variable.is_pointer and variable.is_optional: #double pointer
                body.append(Deallocate(variable))
            else: # anyway there should be a better way to handle this
                arg = variable.clone(name = variable.name, is_pointer = True, is_optional = False)
                body.append(Deallocate(arg))


        if variable.is_optional:
            body.append(FunctionCall(free, [variable]))

        return body

    def free_allocated_memory(self):
        """
        """
        body = []

        for elem in  self.to_free_objects:
            body += self.get_free_statements(elem)

        self.to_free_objects.clear()

        return body

    def need_memory_allocation(self, argument, variable):
        """
        allocated needed memory to hold value this is used to avoid creating mass
        temporary variables and multiples checks, add the allocated variable
        to list (to_free_objects) to free it later
        Parameters:
        -----------
        variable : Variable
            variable to allocate if needed
        Returns     : List of Assign
        -----------
        """
        body = []

        if variable.is_optional:
            if variable.rank > 0:
                dtype = 't_ndarray'
            else:
                dtype = self.find_in_dtype_registry(self._print(variable.dtype), variable.precision)

            size = Variable(NativeGeneric(), dtype)
            body += [AliasAssign(variable,
                FunctionCall(malloc, [
                    FunctionCall(sizeof, [size])
                    ])
                )]
            self.to_free_objects.append(argument)

        return body

    #--------------------------------------------------------------------
    #                   Convert functions
    #--------------------------------------------------------------------

    def generate_converter_function(self, name, cast_function, c_var, is_interface = False):
        """
        """
        c_arg = c_var.clone(name = 'c_arg', is_pointer = True)
        p_arg = Variable(name = 'p_arg', dtype = PyccelPyObject(), is_pointer = True)

        # Setting up the destructor to free allocated memory in case of error
        check = PyccelIs(VariableAddress(p_arg), Nil())
        body = [IfSection(check, self.get_free_statements(c_arg) + [RETURN_ZERO])]

        # Setting up the converter to know if an argument can be optional
        if c_var.is_optional:
            check = PyccelEq(VariableAddress(p_arg), VariableAddress(Py_None))
            body.append(IfSection(check, [Return([Py_CLEANUP_SUPPORTED])]))

        # Getting argument type check statement
        if c_arg.rank > 0:
            check = array_checker(p_arg, c_arg, is_interface, self._target_language)
            body.append(IfSection(check, [RETURN_ZERO]))

        if not is_interface:
            check = PyccelNot(scalar_object_check(p_arg, c_arg, False))
            error = [generate_datatype_error(c_arg)]
            body.append(IfSection(check, error + [RETURN_ZERO]))

        body = [If(*body)]

        # Allocate memory if needed
        body.extend(self.need_memory_allocation(c_var, c_arg))

        # Collect value from python object and check if and error occurred during conversion
        body.append(Assign(c_arg, FunctionCall(cast_function, [p_arg])))
        check = FunctionCall(PyErr_Occurred, [])
        body.append(If(IfSection(check, self.get_free_statements(c_arg) + [RETURN_ZERO])))

        # Succes return
        body.append(Return([Py_CLEANUP_SUPPORTED]))

        function = FunctionDef(name      = name,
                               arguments = [p_arg, c_arg],
                               body      = body,
                               results   = [Variable(name = 'r', dtype = NativeInteger(), is_temp = True)])

        return function

    def generate_converter_function_name(self, used_names, argument):
        """
        Generate an unique name for the converter function
        Parameters:
        ----------
        used_names : Set of strings
            Set of variable and function names to avoid name collisions

        argument   : Variable
            variable holding information needed to choose the converter
            function name

        Returns:
        --------
        name : String
            the generated name

        """
        dtype = self._print(argument.dtype)
        prec  = argument.precision
        dtype = self.find_in_dtype_registry(dtype, prec)
 
        rank   = ''   if argument.rank < 1 else '_{}'.format(argument.rank)       
        order  = ''   if argument.order is None else '_{}'.format(argument.order)
        valued = ''   if not isinstance(argument, ValuedVariable) else 'v_'
        valued = 'o_' if argument.is_optional else valued

        name = 'py_to_{valued}{dtype}{precision}{rank}{order}'.format(
            valued    = valued,
            dtype     = dtype,
            precision = prec,
            rank      = rank,
            order     = order
        )
        name = self.get_new_name(used_names, name) # to avoid name collision
        return name

    # -------------------------------------------------------------------
    #       Parsing arguments and building values functions
    # -------------------------------------------------------------------

    def get_PyArgParse_Converter(self, used_names, argument):
        """
        Responsible for collecting any necessary intermediate functions which are used
        to convert python to C. To avoid creating the same converter, functions are
        stored in a dictionary with a custom key that depend on argument property
        Parameters:
        ----------
        used_names : Set of strings
            Set of variable and function names to avoid name collisions

        argument   : Variable
            variable holding information needed to choose the converter function

        Returns:
        --------
        function : FunctionDef
            the converter function

        """
        key = get_custom_key(argument)

        # Chech if converter already created
        if key in self._converter_functions:
            return self._converter_functions[key]

        name = self.generate_converter_function_name(used_names, argument)

        if argument.rank > 0:
            cast_function = pyarray_to_f_ndarray if self._target_language is 'fortran'\
                                            else pyarray_to_c_ndarray
        else:
            try:
                cast_function = Python_to_C(argument)
            except KeyError:
                raise NotImplementedError(
                'parser not implemented for this datatype : {}'.format(argument.dtype))

        function = self.generate_converter_function(name, cast_function, argument)
        self._converter_functions[key] = function

        return function

    def get_PyBuildValue_Converter(self, result):
        """
        Responsible for collecting any necessary intermediate functions which are used
        to convert c type to python.
        Parameters:
        -----------
        result : Variable
            variable holding information needed to choose the converter function

        Returns:
        --------
        function   : FunctionDef
            the converter function
        """
        # TODO this function should look the same as get_PyArgParse_Converter
        # when returning non scalar datatypes

        if result.rank > 0:
            raise NotImplementedError('return not implemented for arrays.')

        try:
            function = C_to_Python(result)
        except KeyError:
            raise NotImplementedError(
            'return not implemented for this datatype : {}'.format(result.dtype))

        return function

    #--------------------------------------------------------------------
    #                 _print_ClassName functions
    #--------------------------------------------------------------------
    def python_function_as_argument(self, wrapper_name, wrapper_args, wrapper_results):
        """
        Given that we cannot parse function as argument, create a wrapper
        function that raise NotImplemented exception and return NULL
        Parameters:
        -----------
        wrapper_name    : String
            The name of the C wrapper function

        wrapper_args    : List of variables
            List of python object variables

        wrapper_results : Variable
            python object variable

        Returns:
        --------
        String
            return string that contains printed functionDef
        """
        wrapper_func = FunctionDef(
                name      = wrapper_name,
                arguments = wrapper_args,
                results   = wrapper_results,
                body      = [
                                PyErr_SetString('PyExc_NotImplementedError',
                                            '"Cannot pass a function as an argument"'),
                                AliasAssign(wrapper_results[0], Nil()),
                                Return(wrapper_results)
                            ])

        return CCodePrinter._print_FunctionDef(self, wrapper_func)

    def private_function_printer(self, wrapper_name, wrapper_args, wrapper_results):
        """
        Given that private function are not accessible from python, create a wrapper
        function that raise NotImplemented exception and return NULL
        Parameters:
        -----------
        wrapper_name    : String
            The name of the C wrapper function

        wrapper_args    : List of variables
            List of python object variables

        wrapper_results : Variable
            Python object variable

        Returns:
        --------
        String
            return string that contains printed functionDef
        """
        wrapper_func = FunctionDef(
                name      = wrapper_name,
                arguments = wrapper_args,
                results   = wrapper_results,
                body      = [
                                PyErr_SetString('PyExc_NotImplementedError',
                                        '"Private functions are not accessible from python"'),
                                AliasAssign(wrapper_results[0], Nil()),
                                Return(wrapper_results)
                            ])

        return CCodePrinter._print_FunctionDef(self, wrapper_func)

    def _print_PyccelPyObject(self, expr):
        return 'pyobject'

    def _print_PyccelPyArrayObject(self, expr):
        return 'pyarrayobject'
    
    def _print_AliasAssign(self, expr):
        lhs = expr.lhs
        rhs = expr.rhs
        if isinstance(rhs, Variable):
            rhs = VariableAddress(rhs)

        lhs = self._print(lhs.name)
        rhs = self._print(rhs)
        
        if expr.lhs.is_pointer and expr.lhs.is_optional:
            return '*{} = {};'.format(lhs, rhs)

        return '{} = {};'.format(lhs, rhs)

    def _print_Variable(self, expr):
        if expr.is_pointer and expr.is_optional:
            return '(**{})'.format(expr.name)

        return CCodePrinter._print_Variable(self, expr)
    
    def _print_VariableAddress(self, expr):
        variable = expr.variable
        if variable.is_pointer and variable.is_optional:
            return '(*{})'.format(variable.name)

        if not self.stored_in_c_pointer(variable) and variable.rank > 0:
            return '&{}'.format(variable.name)

        return CCodePrinter._print_VariableAddress(self, expr)

    def _print_PyArgKeywords(self, expr):
        arg_names  = ['"{}"'.format(a) for a in expr.arg_names]
        
        arg_names.append(self._print(Nil()))

        arg_names  = ',\n'.join(arg_names)

        code       = 'static char *{name}[] = {{\n{arg_names}\n}};\n'

        return  code.format(name = expr.name, arg_names = arg_names)

    def _print_PyArg_ParseTupleNode(self, expr):
        name    = 'PyArg_ParseTupleAndKeywords'
        pyarg   = expr.pyarg
        pykwarg = expr.pykwarg
        flags   = expr.flags
        # All args are modified so even pointers are passed by address
        args  = ', '.join(['&{}'.format(a.name) for a in expr.args])

        if expr.args:
            code = '{name}({pyarg}, {pykwarg}, "{flags}", {kwlist}, {args})'.format(
                            name    = name,
                            pyarg   = pyarg,
                            pykwarg = pykwarg,
                            flags   = flags,
                            kwlist  = expr.arg_names.name,
                            args    = args)
        else :
            code ='{name}({pyarg}, {pykwarg}, "", {kwlist})'.format(
                    name    = name,
                    pyarg   = pyarg,
                    pykwarg = pykwarg,
                    kwlist  = expr.arg_names.name)

        return code

    def _print_PyBuildValueNode(self, expr):
        name  = 'Py_BuildValue'
        flags = expr.flags

        args  = ', '.join(['&{}, &{}'.format(f.name, a.name)
                            for f, a in zip(expr.converters, expr.args)])

        if expr.args:
            code = '{name}("{flags}", {args})'.format(name  = name,
                                                      flags = flags,
                                                      args  = args)
        else :
            code = '{name}("")'.format(name = name)

        return code

    def _print_Interface(self, expr):
        funcs = expr.functions

        # Save all used names
        used_names = set([a.name for a in funcs[0].arguments]
            + [r.name for r in funcs[0].results]
            + [f.name for f in funcs])

        # Find a name for the wrapper function
        wrapper_name = self.get_wrapper_name(used_names, expr)

        # Collect arguments and results
        wrapper_args    = self.get_wrapper_arguments(used_names)
        wrapper_results = [self.get_new_PyObject("result", used_names)]

        # build keyword_list
        arg_names         = [a.name for a in funcs[0].arguments]
        keyword_list_name = self.get_new_name(used_names, 'kwlist')
        keyword_list      = PyArgKeywords(keyword_list_name, arg_names)

        wrapper_body      = []
        # variable holding the bitset of type check
        check_variable = Variable(dtype = NativeInteger(), precision = 8,
                                   name = self.get_new_name(used_names , "check"))

        # temporary parsing args needed to hold python value
        parse_args     = [self.get_new_PyObject(a.name + 'tmp', used_names)
                          for a in funcs[0].arguments]

        #dict to collect each variable possible type
        types_dict     = OrderedDict((a, list()) for a in parse_args)
        
        # To store the mini function responsible for collecting value and
        # calling interfaces functions and return the builded value
        wrapper_functions           = []

        for func in expr.functions:
            mini_wrapper_name = self.get_wrapper_name(used_names, expr)
            mini_wrapper_body = []
            func_args         = []
            flag              = 0

            # loop on all functions argument to collect needed converter functions
            for c_arg, p_arg in zip(func.arguments, parse_args):
                function = self.get_PyArgParse_Converter(used_names, c_arg)
                func_args.extend(self.get_static_args(c_arg)) # Bind_C args

                flag = self.set_flag_value(flag, c_arg) # set flag value
                types_dict[p_arg].append(c_arg) # collect type

                if isinstance(c_arg, ValuedVariable):
                    mini_wrapper_body.append(self.get_default_assign(c_arg))

                call = FunctionCall(function, [p_arg, VariableAddress(c_arg)]) # convert py to c type
                body = If(IfSection(PyccelNot(call), [RETURN_NULL]))            # check in cas of error
                mini_wrapper_body.append(body)

            # Call function
            static_function = self.get_static_function(func)
            function_call   = FunctionCall(static_function, func_args)

            if len(func.results) > 0:
                results       = func.results if len(func.results) > 1 else func.results[0]
                function_call = Assign(results, function_call)

            mini_wrapper_body.append(function_call)

            # loop on all results to collect needed converter functions
            converters = []
            for res in func.results:
                function = self.get_PyBuildValue_Converter(res)
                converters.append(function)

            # builde results
            build_node = PyBuildValueNode(func.results, converters)

            mini_wrapper_body.append(AliasAssign(wrapper_results[0], build_node))
            # Return
            mini_wrapper_body.append(Return(wrapper_results))
            # Creating mini_wrapper functionDef
            mini_wrapper_function  = FunctionDef(
                    name        = mini_wrapper_name,
                    arguments   = parse_args,
                    results     = wrapper_results,
                    body        = mini_wrapper_body,
                    local_vars  = func.arguments)
            wrapper_functions.append(mini_wrapper_function)

            # call mini_wrapper function from the interface wrapper with the correponding check
            call  = AliasAssign(wrapper_results[0], FunctionCall(mini_wrapper_function, parse_args))
            check = IfSection(PyccelEq(check_variable, LiteralInteger(flag)), [call])
            wrapper_body.append(check)

        # Errors / Types management
        # Creating check_type function
        #check_function = self.generate_interface_check_function(types_dict)
        #wrapper_functions.append(check_function)
        # generate error
        wrapper_body.append(IfSection(LiteralTrue(),
            [PyErr_SetString('PyExc_TypeError', '"Arguments combinations don\'t exist"'),
             RETURN_NULL]))

        wrapper_body = [If(*wrapper_body)]
        # Parse arguments
        parse_node = PyArg_ParseTupleNode(*wrapper_args[1:], keyword_list,
                                           funcs[0].arguments, parse_args = parse_args)

        parse_node   = If(IfSection(PyccelNot(parse_node), [RETURN_NULL]))
        #check_call   = Assign(check_variable, FunctionCall(check_function, parse_args))
        wrapper_body = [keyword_list, parse_node] + wrapper_body

        # Create FunctionDef for interface wrapper
        wrapper_functions.append(FunctionDef(
                    name       = wrapper_name,
                    arguments  = wrapper_args,
                    results    = wrapper_results,
                    body       = wrapper_body,
                    local_vars = parse_args + [check_variable]))

        sep = self._print(SeparatorComment(40))

        return sep + '\n'.join(CCodePrinter._print_FunctionDef(self, f) for f in wrapper_functions)

    def _print_FunctionDef(self, expr):
        # Save all used names
        used_names = set([a.name for a in expr.arguments]
                    + [r.name for r in expr.results]
                    + [expr.name])

        # Find a name for the wrapper function
        wrapper_name    = self.get_wrapper_name(used_names, expr)

        # Collect arguments and results
        wrapper_args    = self.get_wrapper_arguments(used_names)
        wrapper_results = [self.get_new_PyObject("result", used_names)]

        # Build keyword_list
        arg_names         = [a.name for a in expr.arguments]
        keyword_list_name = self.get_new_name(used_names, 'kwlist')
        keyword_list      = PyArgKeywords(keyword_list_name, arg_names)

        wrapper_body      = [keyword_list]
        static_func_args  = []

        if expr.is_private:
            return self.private_function_printer(wrapper_name, wrapper_args, wrapper_results)

        if any(isinstance(arg, FunctionAddress) for arg in expr.arguments):
            return self.python_function_as_argument(wrapper_name, wrapper_args, wrapper_results)

        converters = []
        # Loop on all the arguments and collect the needed converter functions
        for c_arg in expr.arguments:
            function = self.get_PyArgParse_Converter(used_names, c_arg)
            converters.append(function)

            # Get Bind/C arguments
            static_func_args.extend(self.get_static_args(c_arg))

            # Set default value when the argument is valued
            if isinstance(c_arg, ValuedVariable):
                wrapper_body.append(self.get_default_assign(c_arg))

        # Parse arguments
        parse_node = PyArg_ParseTupleNode(*wrapper_args[1:], keyword_list, expr.arguments,
                                          converters = converters)

        wrapper_body.append(If(IfSection(PyccelNot(parse_node), [RETURN_NULL])))

        # Call function
        static_function = self.get_static_function(expr)
        function_call   = FunctionCall(static_function, static_func_args)

        if len(expr.results) > 0:
            results       = expr.results if len(expr.results) > 1 else expr.results[0]
            function_call = Assign(results, function_call)

        wrapper_body.append(function_call)

        # Loop on all the results and collect the needed converter functions
        converters = []
        for res in expr.results:
            function = self.get_PyBuildValue_Converter(res)
            converters.append(function)

        # Builde results
        build_node = PyBuildValueNode(expr.results, converters)

        wrapper_body.append(AliasAssign(wrapper_results[0], build_node))

        # Free all allocated memory :
        wrapper_body.extend(self.free_allocated_memory())
        
        # Return
        wrapper_body.append(Return(wrapper_results))

        wrapper_function = FunctionDef(name        = wrapper_name,
                                       arguments   = wrapper_args,
                                       results     = wrapper_results,
                                       body        = wrapper_body,
                                       local_vars  = expr.arguments + expr.results)

        return CCodePrinter._print_FunctionDef(self, wrapper_function)

    def _print_Module(self, expr):
        self._global_names = set(f.name for f in expr.funcs)
        self._module_name  = expr.name

        static_funcs = [self.get_static_function(func) for func in expr.funcs]

        function_signatures = '\n'.join('{};'.format(self.static_function_signature(f))
                                                        for f in static_funcs)

        interface_funcs = [f.name for i in expr.interfaces for f in i.functions]
        funcs = [*expr.interfaces, *(f for f in expr.funcs if f.name not in interface_funcs)]

        function_defs         = '\n\n'.join(self._print(f) for f in funcs)

        converters            = '\n\n'.join(CCodePrinter._print_FunctionDef(self, c)
                                for c in self._converter_functions.values())

        method_def_func = ',\n'.join(('{{\n"{name}",\n'
                                    '(PyCFunction){wrapper_name},\n'
                                    'METH_VARARGS | METH_KEYWORDS,\n'
                                    '{doc_string}\n'
                                    '}}').format(
                                            name         = f.name,
                                            wrapper_name = self._function_wrapper_names[f.name],
                                            doc_string   = self._print(LiteralString('\n'.join(f.doc_string.comments)))\
                                                            if f.doc_string else '""') for f in funcs)

        method_def_name = self.get_new_name(self._global_names, '{}_methods'.format(expr.name))

        method_def = ('static PyMethodDef {method_def_name}[] = {{\n'
                            '{method_def_func},\n'
                            '{{ NULL, NULL, 0, NULL}}\n'
                            '}};'.format(method_def_name = method_def_name,
                                        method_def_func = method_def_func))

        module_def_name = self.get_new_name(self._global_names, '{}_module'.format(expr.name))
        module_def = ('static struct PyModuleDef {module_def_name} = {{\n'
                    'PyModuleDef_HEAD_INIT,\n'
                    '/* name of module */\n'
                    '\"{mod_name}\",\n'
                    '/* module documentation, may be NULL */\n'
                    'NULL,\n' #TODO: Add documentation
                    '/* size of per-interpreter state of the module, or -1'
                    'if the module keeps state in global variables. */\n'
                    '-1,\n'
                    '{method_def_name}\n'
                    '}};'.format(module_def_name = module_def_name,
                                mod_name        = expr.name,
                                method_def_name = method_def_name))


        init_func = ('PyMODINIT_FUNC PyInit_{mod_name}(void)\n{{\n'
                    'PyObject *m;\n\n'
                    'import_array();\n\n'
                    'm = PyModule_Create(&{module_def_name});\n'
                    'if (m == NULL) return NULL;\n\n'
                    'return m;\n}}'.format(mod_name        = expr.name,
                                        module_def_name = module_def_name))

        # Print imports last to be sure that all additional_imports have been collected
        imports  = [Import(s) for s in self._additional_imports]
        imports += [Import('numpy/arrayobject')]
        imports += [Import('cwrapper')]
        imports  = '\n'.join(self._print(i) for i in imports)

        sep = self._print(SeparatorComment(40))

        return ('#define PY_ARRAY_UNIQUE_SYMBOL CWRAPPER_ARRAY_API\n'
                '{imports}\n\n'
                '{function_signatures}\n\n'
                '{sep}\n\n'
                '{converters}\n\n'
                '{sep}\n\n'
                '{function_defs}\n\n'
                '{method_def}\n\n'
                '{sep}\n\n'
                '{module_def}\n\n'
                '{sep}\n\n'
                '{init_func}\n'.format(
                    imports              = imports,
                    function_signatures  = function_signatures,
                    sep                  = sep,
                    converters           = converters,
                    function_defs        = function_defs,
                    method_def           = method_def,
                    module_def           = module_def,
                    init_func            = init_func))

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
    #TODO is this docstring upto date ?
    return CWrapperCodePrinter(parser, target_language, **settings).doprint(expr, assign_to)
