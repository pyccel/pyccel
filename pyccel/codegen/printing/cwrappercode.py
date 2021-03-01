# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=R0201

import numpy as np

from pyccel.codegen.printing.ccode import CCodePrinter

from pyccel.ast.literals    import LiteralTrue, LiteralInteger, LiteralString

from pyccel.ast.operators   import PyccelNot, PyccelEq, PyccelOr
from pyccel.ast.datatypes   import NativeBool, NativeComplex, NativeInteger
from pyccel.ast.core        import create_incremented_string, SeparatorComment

from pyccel.ast.core        import FunctionCall, FunctionDef, FunctionAddress
from pyccel.ast.core        import Assign, AliasAssign, Nil
from pyccel.ast.core        import If, IfSection, Import, Return

from pyccel.ast.cwrapper    import (PyArgKeywords, PyArg_ParseTupleNode,
                                    PyBuildValueNode)
from pyccel.ast.cwrapper    import Python_to_C, C_to_Python, PythonType_Check
from pyccel.ast.cwrapper    import get_custom_key, flags_registry

from pyccel.ast.cwrapper    import PyccelPyObject, PyccelPyArrayObject, Py_None
from pyccel.ast.cwrapper    import get_custom_key, PyErr_SetString, PyErr_Occurred

from pyccel.ast.numpy_wrapper   import Check_Array, NumpyType_Check, PyArray_to_Array

from pyccel.ast.internals       import PyccelArraySize
from pyccel.ast.variable        import Variable, ValuedVariable, VariableAddress

from pyccel.ast.bind_c          import as_static_function_call

from pyccel.errors.errors   import Errors
from pyccel.errors.messages import PYCCEL_RESTRICTION_TODO

errors = Errors()

__all__ = ["CWrapperCodePrinter", "cwrappercode"]

dtype_registry = {('pyobject'     , 0) : 'PyObject',
                  ('pyarrayobject', 0) : 'PyArrayObject'}

class CWrapperCodePrinter(CCodePrinter):
    """A printer to convert a python module to strings of c code creating
    an interface between python and an implementation of the module in c"""
    def __init__(self, parser, target_language, **settings):
        CCodePrinter.__init__(self, parser, **settings)
        self._target_language             = target_language
        self._function_wrapper_names      = dict()
        self._global_names                = set()
        self._module_name                 = None
        self.converter_functions_dict     = {}

    # --------------------------------------------------------------------
    #                       Helper functions
    # --------------------------------------------------------------------
    @staticmethod
    def get_new_name(used_names, requested_name):
        """
        Generate a new name, return the requested_name if it's not in 
        used_names set  or generate new one based on the requested_name.
        the generated name is appended to the set
        Parameters
        ----------
        used_names     : set of strings
            Set of variable and function names to avoid name collisions
        requested_name : str
            The desired name
        Returns
        ----------
        name  : str

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
        name       : string
            The desired name
        used_names : set of strings
            Set of variable and function names to avoid name collisions
        Returns: Variable
        -------
        """
        return Variable(dtype      = PyccelPyObject(),
                        name       = self.get_new_name(used_names, name),
                        is_pointer = True)


    def get_wrapper_arguments(self, used_names):
        """
        Create wrapper arguments
        Parameters:
        -----------
        used_names : set of strings
            Set of variable and function names to avoid name collisions
        Returns: list of Variable
        """
        python_func_args    = self.get_new_PyObject("args"  , used_names)
        python_func_kwargs  = self.get_new_PyObject("kwargs", used_names)
        python_func_selfarg = self.get_new_PyObject("self"  , used_names)

        return [python_func_selfarg, python_func_args, python_func_kwargs]

    def find_in_dtype_registry(self, dtype, prec):
        """
        """
        try :
            return dtype_registry[(dtype, prec)]
        except KeyError:
            return CCodePrinter.find_in_dtype_registry(self, dtype, prec)

    def get_static_function(self, function):
        """
        Create functionDef for arguments rank > 0 in fortran.
        Parameters:
        -----------
        function    : FunctionDef
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
        func(a) ==> static_func(a.DIM , a.DATA)
        where a.DATA = buffer holding data
              a.DIM = size of array
        Parameters:
        -----------
        argument    : Variable
        Returns     : List of arguments
        -----------
        """
        if self._target_language == 'fortran' and argument.rank > 0:
            return [PyccelArraySize(argument, i) for i in range(argument.rank)] + [argument]
        else:
            return [argument]

    @staticmethod
    def get_flag_value(flag, variable):
        """
        """
        try :
            new_flag = flags_registry[(variable.dtype, variable.precision)]
        except KeyError:
            raise NotImplementedError(
            'datatype not implemented as arguments : {}'.format(variable.dtype))
        return (flag << 4) + flag

    # --------------------------------------------------------------------
    #                  Custom body generators [helpers]
    # --------------------------------------------------------------------
    @staticmethod
    def generate_valued_variable_body(py_variable, c_variable):
        """
        Generate valued variable code section (check, collect default value)
        Parameters:
        ----------
        py_object : Variable
            The python argument needed for check
        c_variable : Variable
            The variable that will hold default value
        Returns   :
        -----------
        body      : IfSection
        """

        check = PyccelEq(VariableAddress(py_variable), VariableAddress(Py_None))
        if c_variable.is_optional:
            body =  [AliasAssign(variable, Nil())]

        else:
            body = [Assign(c_variable, variable.value)]

        body = IfSection(check, body)
        return body
        #TODO modify it to accepte multiple variables or list of variables ?

    # --------------------------------------------------------------------
    #                        Custom error generators
    # --------------------------------------------------------------------
    @staticmethod
    def generate_datatype_error(variable):
        """
        Generate TypeError exception from the variable information (datatype, precision)
        Parameters:
        ----------
        variable : Variable

        Returns:
        -------
        func     : FunctionCall
            call to PyErr_SetString with TypeError as exception and custom message
        """
        dtype     = variable.dtype

        if isinstance(dtype, NativeBool):
            precision = ''
        if isinstance(dtype, NativeComplex):
            precision = '{} bit '.format(variable.precision * 2 * 8)
        else:
            precision = '{} bit '.format(variable.precision * 8)

        message = '"Argument must be {precision}{dtype}"'.format(
                precision = precision,
                dtype     = variable.dtype)
        return PyErr_SetString('PyExc_TypeError', message)

    #--------------------------------------------------------------------
    #                   Convert functions
    #--------------------------------------------------------------------

    def generate_scalar_converter_function(self, used_names, variable, check_is_needed = True):
        """
        Generate converter function responsible for collecting value 
        and managing errors (data type, precision) of arguments
        with rank less than 1
        Parameters:
        ----------
        used_names : set of strings
            Set of variable and function names to avoid name collisions
        variable   : Variable
            variable holdding information (data type, precision) needed
            in bulding converter function body 
        check_is_needed : Boolean
            True if data type check is needed, used to avoid multiple type check
            in interface
        Returns:
        --------
        funcDef   : FunctionDef
        """

        func_name       = 'py_to_{}'.format(self._print(variable.dtype))
        func_name       = self.get_new_name(used_names, func_name)
        py_variable     = Variable(name = 'py_variable', dtype = PyccelPyObject(), is_pointer = True)
        c_variable      = variable.clone(name = 'c', is_pointer = True)
        body            = []

        # (Valued / Optional) variable check
        if isinstance(variable, ValuedVariable):
            body.append(self.generate_valued_variable_body(py_variable, c_variable))

        #datatqype check
        if check_is_needed:
            numpy_check  = NumpyType_Check(py_variable, c_variable)
            python_check = PythonType_Check(py_variable, c_variable)
            check        = PyccelNot(PyccelOr(numpy_check, python_check))
            error        = self.generate_datatype_error(c_variable)
            body.append(IfSection(check, [error, Return([LiteralInteger(0)])]))

        body = [If(*body)]
        # Collect value
        body.append(Assign(c_variable, FunctionCall(Python_to_C(c_variable), [py_variable])))
        # call PyErr_Occurred to check any error durring conversion
        body.append(If(IfSection(FunctionCall(PyErr_Occurred, []), [Return([LiteralInteger(0)])])))
        body.append(Return([LiteralInteger(1)]))

        funcDef = FunctionDef(name     = func_name,
                            arguments  = [py_variable, c_variable],
                            results    = [],
                            body       = body)

        return funcDef

    def generate_array_converter_function(self, used_names, variable, check_is_needed = True):
        """
        Generate converter function responsible for collecting value 
        and managing errors (data type, rank, order) of arguments
        with rank greater than 0
        Parameters:
        ----------
        used_names : set of strings
            Set of variable and function names to avoid name collisions
        variable   : Variable
            variable holdding information (data type, rank, order) needed
            in bulding converter function body 
        check_is_needed : Boolean
            True if data type check is needed, used to avoid multiple type check
            in interface
        Returns:
        --------
        funcDef   : FunctionDef
        """

        func_name       = 'py_to_nd{}'.format(self._print(variable.dtype))
        func_name       = self.get_new_name(used_names, func_name)
        py_variable     = Variable(name = 'py_variable', dtype = PyccelPyObject(), is_pointer = True)
        py_array        = Variable(name = 'py_array', dtype = PyccelPyArrayObject(), is_pointer = True)
        c_variable      = variable.clone(name = 'c_variable', is_pointer = True)
        body            = []

        # (Valued / Optional) variable check
        if isinstance(variable, ValuedVariable):
            body.append(If(self.generate_valued_variable_body(py_object, c_variable)))

        #array check
        body.append(AliasAssign(py_array, Check_Array(py_variable, c_variable, self._target_language)))
        body.append(If(IfSection(
            PyccelNot(VariableAddress(py_array)),
            [Return([LiteralInteger(0)])])))

        #datatqype check
        if check_is_needed:
            check = PyccelNot(NumpyType_Check(py_array, c_variable))
            error = self.generate_datatype_error(c_variable)
            body.append(If(IfSection(check, [error, Return([LiteralInteger(0)])])))

        # Convert numpy_array to c array
        body.append(Assign(c_variable, FunctionCall(PyArray_to_Array, [py_array])))
        body.append(Return([LiteralInteger(1)]))

        funcDef = FunctionDef(name     = func_name,
                            arguments  = [py_variable, c_variable],
                            results    = [],
                            local_vars = [py_array],
                            body       = body)

        return funcDef

    def generate_tuple_converter_function(self, used_names, variable):
        pass
        #TODO

    def generate_list_converter_function(self, used_names, variable):
        pass
        #TODO

    # -------------------------------------------------------------------
    #       Parsing arguments and building values  functions
    # -------------------------------------------------------------------

    def get_PyArgParse_Converter_Function(self, used_names, variable, check_is_needed = True):
        """
        Responsible for collecting any necessary intermediate functions which are used
        to convert python to C.
        Parameters:
        ----------
        used_names : set of strings
            Set of variable and function names to avoid name collisions
        variable   : Variable
            variable holding information needed to chose converter function
            in bulding converter function body
        check_is_needed : Boolean
            True if data type check is needed, used to avoid multiple type check
            in interface
        Returns:
        --------
        function : FunctionDef
        """
        if get_custom_key(variable) not in self.converter_functions_dict:

            if variable.rank > 0:
                function = self.generate_array_converter_function(used_names, variable)
            else:
                function = self.generate_scalar_converter_function(used_names, variable)

            self.converter_functions_dict[get_custom_key(variable)] = function

            return function
        return self.converter_functions_dict[get_custom_key(variable)]

    def get_PyBuildValue_Converter_function(self, variable):
        """
        Responsible for collecting any necessary intermediate functions which are used
        to convert c type to python.
        Parameters:
        ----------
        variable : Variable
            variable holding information needed to chose converter function
        Returns:
        --------
        func     : FunctionDef
        """
        if variable.rank > 0:
            raise NotImplementedError('return not implemented for arrays.')

        try:
            func = C_to_Python(variable)
        except KeyError:
            raise NotImplementedError(
            'return not implemented for this datatype : {}'.format(variable.dtype))
        
        return func

    #--------------------------------------------------------------------
    #                 _print_ClassName functions
    #--------------------------------------------------------------------

    def _print_PyccelPyObject(self, expr):
        return 'pyobject'

    def _print_PyccelPyArrayObject(self, expr):
        return 'pyarrayobject'

    def _print_PyArgKeywords(self, expr):
        arg_names  = ['"{}"'.format(a) for a in expr.arg_names]
        arg_names += [self._print(Nil())]
        
        arg_names  = ',\n'.join(arg_names)
        code       = 'static char *{name}[] = {{\n{arg_names}\n}};\n'

        return  code.format(name = expr.name, arg_names = arg_names)

    def _print_PyArg_ParseTupleNode(self, expr):
        name    = 'PyArg_ParseTupleAndKeywords'
        pyarg   = expr.pyarg
        pykwarg = expr.pykwarg
        flags   = expr.flags
        # All args are modified so even pointers are passed by address
        args    = ', '.join(['&{}'.format(a.name) for a in expr.args])

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
        args  = ', '.join(['&{}'.format(a.name) for a in expr.args])

        #to change for args rank 1 +
        if expr.args:
            code = '{name}("{flags}", {args})'.format(name = name,
                                                     flags = flags,
                                                     args  = args)
        else :
            code = '{name}("")'.format(name = name)

        return code


    def _print_Interface(self, expr):
        funcs = expr.functions

        # Save all used names
        used_names = set([a.name for a in expr.func[0].arguments]
            + [r.name for r in expr.func[0].results]
            + [f.name for f in expr.func])

        # Find a name for the wrapper function
        wrapper_name = self.get_wrapper_name(used_names, expr)

        # Collect arguments and results
        wrapper_args    = self.get_wrapper_arguments(used_names)
        wrapper_results = [self.get_new_PyObject("result", used_names)]

        # build keyword_list
        arg_names         = [a.name for a in expr.arguments]
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
        types_dict     = OrderedDict((a, set()) for a in funcs[0].arguments)
        # To store the mini function responsible for collecting value and
        # calling interfaces functions and return the builded value
        wrapper_functions = []
        for func in expr.functions:
            mini_wrapper_name = self.get_wrapper_name(used_names, expr)
            mini_wrapper_body = []
            func_args         = []
            flag              = 0

            # loop on all functions argument to collect needed converter functions
            for f_arg, p_arg in zip(func.arguments, parse_args):
                convert_func = self.get_PyArgParse_Converter_Function(used_names, f_arg)
                func_args.extend(self.get_static_args(arg)) # Bind_C args

                flag = self.get_flag_value(flag, f_arg) # set flag value
                types_dict[p_arg].add(f_arg.dtype) # collect type
                call = FunctionCall(p_arg, VariableAddress(f_arg)) # convert py to c type
                body = If(IfSection(PyccelNot(call), Return([Nil()]))) # check in cas of error
                mini_wrapper_body.append(body)

            # Call function
            static_function = self.get_static_function(func)
            function_call   = FunctionCall(static_function, func_args)

            if len(func.results) > 0:
                results       = func.results if len(func.results) > 1 else func.results[0]
                function_call = Assign(results, function_call)

            mini_wrapper_body.append(function_call)

            # loop on all results to collect needed converter functions
            building_converter_functions = {}
            for res in expr.results:
                convert_func = self.get_PyBuildValue_Converter_function(res)
            building_converter_functions[get_custom_key(res)] = convert_func

            # builde results
            build_node = PyBuildValueNode(func.results, building_converter_functions)

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
            check = If(IfSection(PyccelEq(check_variable, LiteralInteger(flags)), [call]))
            wrapper_body.append(check)

        # Errors / Types management
        # Creating check_type function
        check_function = self.generate_interface_check_function(types_dict)
        wrapper_functions.append(check_function)
        # generate error
        wrapper_body.append(IfSection(LiteralTrue(),
            [PyErr_SetString('PyExc_TypeError', '"Arguments combinations don\'t exist"'),
             Return([Nil()])]))

        # Parse arguments
        parse_node = PyArg_ParseTupleNode(*wrapper_args[:-1],
                                    self.converter_functions_dict,
                                    expr.arguments, keyword_list)

        parse_node   = If(IfSection(PyccelNot(parse_node), [Return([Nil()])]))
        check_call   = Assign(check_variable, FunctionCall(check_function, parse_args))
        wrapper_body = [keyword_list, parse_node, check_call] + wrapper_body

        # Create FunctionDef for interface wrapper
        funcs_def.append(FunctionDef(
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
        wrapper_name = self.get_wrapper_name(used_names, expr)

        # Collect arguments and results
        wrapper_args    = self.get_wrapper_arguments(used_names)
        wrapper_results = [self.get_new_PyObject("result", used_names)]

        # build keyword_list
        arg_names         = [a.name for a in expr.arguments]
        keyword_list_name = self.get_new_name(used_names, 'kwlist')
        keyword_list      = PyArgKeywords(keyword_list_name, arg_names)

        wrapper_body      = [keyword_list]
        func_args         = []
        # loop on all functions argument to collect needed converter functions
        for arg in expr.arguments:
            self.get_PyArgParse_Converter_Function(used_names, arg)
            func_args.extend(self.get_static_args(arg)) # Bind_C args

        # Parse arguments
        parse_node = PyArg_ParseTupleNode(*wrapper_args[:-1],
                                          self.converter_functions_dict,
                                          expr.arguments, keyword_list)

        wrapper_body.append(If(IfSection(PyccelNot(parse_node), [Return([Nil()])])))

        # Call function
        static_function = self.get_static_function(expr)
        function_call   = FunctionCall(static_function, func_args)

        if len(expr.results) > 0:
            results       = expr.results if len(expr.results) > 1 else expr.results[0]
            function_call = Assign(results, function_call)

        wrapper_body.append(function_call)

        # loop on all results to collect needed converter functions
        building_converter_functions = {}
        for res in expr.results:
            convert_func = self.get_PyBuildValue_Converter_function(res)
            building_converter_functions[get_custom_key(res)] = convert_func

        # builde results
        build_node = PyBuildValueNode(expr.results, building_converter_functions)

        wrapper_body.append(AliasAssign(wrapper_results[0], build_node))
        # Return
        wrapper_body.append(Return(wrapper_results))
        wrapper_function = FunctionDef(name     = wrapper_name,
                                    arguments   = wrapper_args,
                                    results     = wrapper_results,
                                    body        = wrapper_body,
                                    local_vars = tuple(expr.arguments + expr.results))
        
        return CCodePrinter._print_FunctionDef(self, wrapper_function)

    def _print_Module(self, expr):
        self._global_names = set(f.name for f in expr.funcs)
        self._module_name  = expr.name
        
        static_funcs = [self.get_static_function(func) for func in expr.funcs]
        
        function_signatures = '\n'.join('{};'.format(self.function_signature(f)) for f in static_funcs)

        interface_funcs = [f.name for i in expr.interfaces for f in i.functions]
        funcs = [*expr.interfaces, *(f for f in expr.funcs if f.name not in interface_funcs)]

        function_defs         = '\n\n'.join(self._print(f) for f in funcs)
        converters_functions  = '\n\n'.join(CCodePrinter._print_FunctionDef(self, f)
                                    for f in self.converter_functions_dict.values())

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
        imports += [Import('cwrapper')]
        imports  = '\n'.join(self._print(i) for i in imports)

        numpy_max_acceptable_version = [1, 19]
        numpy_current_version = [int(v) for v in np.version.version.split('.')[:2]]
        numpy_api_macro = '#define NPY_NO_DEPRECATED_API NPY_{}_{}_API_VERSION'.format(
                min(numpy_max_acceptable_version[0], numpy_current_version[0]),
                min(numpy_max_acceptable_version[1], numpy_current_version[1]))
        
        sep = self._print(SeparatorComment(40))

        return ('#define PY_SSIZE_T_CLEAN\n'
                '{numpy_api_macro}\n'
                '{imports}\n\n'
                '{function_signatures}\n\n'
                '{sep}\n\n'
                '{converters_functions}\n\n'
                '{sep}\n\n'
                '{function_defs}\n\n'
                '{method_def}\n\n'
                '{sep}\n\n'
                '{module_def}\n\n'
                '{sep}\n\n'
                '{init_func}\n'.format(
                    numpy_api_macro      = numpy_api_macro,
                    imports              = imports,
                    function_signatures  = function_signatures,
                    sep                  = sep,
                    converters_functions = converters_functions,
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
