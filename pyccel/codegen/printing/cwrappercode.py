# coding: utf-8
# pylint: disable=R0201

from collections import OrderedDict

import numpy as np

from pyccel.codegen.printing.ccode import CCodePrinter

from pyccel.ast.literals  import LiteralTrue, LiteralInteger

from pyccel.ast.builtins import PythonPrint

from pyccel.ast.core import Variable, ValuedVariable, Assign, AliasAssign, FunctionDef, FunctionAddress
from pyccel.ast.core import If, Nil, Return, FunctionCall
from pyccel.ast.core import create_incremented_string, SeparatorComment
from pyccel.ast.core import VariableAddress, Import, IfTernaryOperator

from pyccel.ast.operators import PyccelEq, PyccelNe, PyccelNot

from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeComplex, NativeReal

from pyccel.ast.cwrapper import PyccelPyObject, PyArg_ParseTupleNode, PyBuildValueNode
from pyccel.ast.cwrapper import PyArgKeywords, collect_function_registry
from pyccel.ast.cwrapper import Py_None
from pyccel.ast.cwrapper import PyErr_SetString, PythonType_Check
from pyccel.ast.cwrapper import cast_function_registry, Py_DECREF
from pyccel.ast.cwrapper import PyccelPyArrayObject
from pyccel.ast.cwrapper import numpy_get_ndims, numpy_get_data, numpy_get_dim
from pyccel.ast.cwrapper import numpy_get_type, numpy_dtype_registry
from pyccel.ast.cwrapper import numpy_check_flag, numpy_flag_c_contig, numpy_flag_f_contig

from pyccel.ast.bind_c   import as_static_function_call

from pyccel.errors.errors import Errors
from pyccel.errors.messages import PYCCEL_RESTRICTION_TODO

errors = Errors()

__all__ = ["CWrapperCodePrinter", "cwrappercode"]

dtype_registry = {('pyobject'     , 0) : 'PyObject',
                  ('pyarrayobject', 0) : 'PyArrayObject'}

class CWrapperCodePrinter(CCodePrinter):
    """A printer to convert a python module to strings of c code creating
    an interface between python and an implementation of the module in c"""
    def __init__(self, parser, target_language, settings=None):
        CCodePrinter.__init__(self, parser,settings)
        self._target_language = target_language
        self._cast_functions_dict = OrderedDict()
        self._to_free_PyObject_list = []
        self._function_wrapper_names = dict()
        self._global_names = set()
        self._module_name = None

    def stored_in_c_pointer(self, a):
        stored_in_c = CCodePrinter.stored_in_c_pointer(self, a)
        if self._target_language == 'fortran':
            return stored_in_c or (isinstance(a, Variable) and a.rank>0)
        else:
            return stored_in_c

    def get_new_name(self, used_names, requested_name):
        if requested_name not in used_names:
            used_names.add(requested_name)
            return requested_name
        else:
            incremented_name, _ = create_incremented_string(used_names, prefix=requested_name)
            return incremented_name

    def function_signature(self, expr):
        args = list(expr.arguments)
        if any([isinstance(a, FunctionAddress) for a in args]):
            # Functions with function addresses as arguments cannot be
            # exposed to python so there is no need to print their signature
            return ''
        else:
            return CCodePrinter.function_signature(self, expr)

    def get_declare_type(self, expr):
        dtype = self._print(expr.dtype)
        prec  = expr.precision
        dtype = self.find_in_dtype_registry(dtype, prec)

        if self.stored_in_c_pointer(expr):
            return '{0} *'.format(dtype)
        else:
            return '{0} '.format(dtype)

    def get_new_PyObject(self, name, used_names):
        return Variable(dtype=PyccelPyObject(),
                        name=self.get_new_name(used_names, name),
                        is_pointer=True)

    def find_in_dtype_registry(self, dtype, prec):
        try :
            return dtype_registry[(dtype, prec)]
        except KeyError:
            return CCodePrinter.find_in_dtype_registry(self, dtype, prec)

    def find_in_numpy_dtype_registry(self, var):
        """ Find the numpy dtype key for a given variable
        """
        dtype = self._print(var.dtype)
        prec  = var.precision
        try :
            return numpy_dtype_registry[(dtype, prec)]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO,
                    symbol = "{}[kind = {}]".format(dtype, prec),
                    severity='fatal')

    def get_default_assign(self, arg, func_arg):
        if arg.rank > 0 :
            return AliasAssign(arg, Nil())
        elif func_arg.is_optional:
            return AliasAssign(arg, Py_None)
        elif isinstance(arg.dtype, (NativeReal, NativeInteger, NativeBool)):
            return Assign(arg, func_arg.value)
        elif isinstance(arg.dtype, PyccelPyObject):
            return AliasAssign(arg, Py_None)
        else:
            raise NotImplementedError('Default values are not implemented for this datatype : {}'.format(func_arg.dtype))

    def _get_static_function(self, used_names, function, collect_dict):
        """
        Create arguments and functioncall for arguments rank > 0 in fortran.
        Format : a is numpy array
        func(a) ==> static_func(a.DIM , a.DATA)
        where a.DATA = buffer holding data
              a.DIM = size of array
        """
        additional_body = []
        if self._target_language == 'fortran':
            static_args = []
            for a in function.arguments:
                if isinstance(a, Variable) and a.rank>0:
                    # Add shape arguments for static function
                    for i in range(collect_dict[a].rank):
                        var = Variable(dtype=NativeInteger() ,name = self.get_new_name(used_names, a.name + "_dim"))
                        body = FunctionCall(numpy_get_dim, [collect_dict[a], i])
                        if a.is_optional:
                            body = IfTernaryOperator(VariableAddress(collect_dict[a]), body , LiteralInteger(0))
                        body = Assign(var, body)
                        additional_body.append(body)
                        static_args.append(var)
                static_args.append(a)
            static_function = as_static_function_call(function, self._module_name, name=function.name)
        else:
            static_function = function
            static_args = function.arguments
        return static_function, static_args, additional_body

    # -------------------------------------------------------------------
    # Functions that take care of creating cast or convert type function call :
    # -------------------------------------------------------------------
    def get_collect_function_call(self, variable, collect_var):
        """
        Represents a call to cast function responsible of collecting value from python object.

        Parameters:
        ----------
        variable: variable
            the variable needed to collect
        collect_var :
            the pyobject variable
        """
        if variable.rank > 0 :
            return FunctionCall(numpy_get_data,[collect_var])

        if isinstance(variable.dtype, NativeComplex):
            return self.get_cast_function_call('pycomplex_to_complex', collect_var)

        if isinstance(variable.dtype, NativeBool):
            return self.get_cast_function_call('pybool_to_bool', collect_var)
        try :
            collect_function = collect_function_registry[variable.dtype]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=variable.dtype,severity='fatal')
        return FunctionCall(collect_function, [collect_var])


    def get_cast_function_call(self, cast_type, arg):
        """
        Represents a call to cast function responsible of the conversion of one data type into another.

        Parameters:
        ----------
        cast_type: string
            The type of cast function on format 'data type_to_data type'
        arg: variable
            the variable needed to cast
        """

        if cast_type in self._cast_functions_dict:
            cast_function = self._cast_functions_dict[cast_type]

        else:
            cast_function_name = self.get_new_name(self._global_names, cast_type)

            try:
                cast_function = cast_function_registry[cast_type](cast_function_name)
            except KeyError as e:
                raise NotImplementedError("No conversion function : {}".format(cast_type)) from e

            self._cast_functions_dict[cast_type] = cast_function

        return FunctionCall(cast_function, [arg])

    # -------------------------------------------------------------------
    # Functions managing  the creation of wrapper body
    # -------------------------------------------------------------------
    def _body_optional_variable(self, tmp_variable, variable, collect_var, check_type = False):
        """
        Responsible for collecting value and managing error and create the body
        of optional arguments in format
                if (pyobject == Py_None){
                    collect Null
                }else if(Type Check == False){
                    Print TypeError Wrong Type
                    return Null
                }else{
                    assign pyobject value to tmp variable
                    collect the adress of the tmp variable
                }
        Parameters:
        ----------
        tmp_variable : Variable
            The temporary variable  to hold result
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
        body = [(PyccelEq(VariableAddress(collect_var), VariableAddress(Py_None)),
                [Assign(VariableAddress(variable), Nil())])]
        if check_type : # Type check
            check = PyccelNot(PythonType_Check(variable, collect_var))
            error = PyErr_SetString('PyExc_TypeError', '"{} must be {}"'.format(variable, variable.dtype))
            body += [(check, [error, Return([Nil()])])]
        body += [(LiteralTrue(), [Assign(tmp_variable, self.get_collect_function_call(variable, collect_var)),
                    Assign(VariableAddress(variable), VariableAddress(tmp_variable))])]
        body = [If(*body)]

        return body

    def _body_valued_variable(self, variable, collect_var, check_type = False) :
        """
        Responsible for collecting value and managing error and create the body
        of valued arguments in format
                if (pyobject == Py_None){
                    collect default value
                }else if(Type Check == False){
                    Print TypeError Wrong Type
                    return Null
                }else{
                    collect the value from PyObject
                }
        Parameters:
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
        body = [(PyccelEq(VariableAddress(collect_var), VariableAddress(Py_None)),
                [Assign(variable, variable.value)])]
        if check_type : # Type check
            check = PyccelNot(PythonType_Check(variable, collect_var))
            error = PyErr_SetString('PyExc_TypeError', '"{} must be {}"'.format(variable, variable.dtype))
            body += [(check, [error, Return([Nil()])])]
        body += [(LiteralTrue(), [Assign(variable, self.get_collect_function_call(variable, collect_var))])]
        body = [If(*body)]

        return body

    def _body_array(self, variable, collect_var, check_type = False) :
        """
        Responsible for collecting value and managing error and create the body
        of arguments with rank greater than 0 in format
                if (rank check == False){
                    print TypeError Wrong rank
                    return Null
                }else if(Type Check == False){
                    Print TypeError Wrong type
                    return Null
                }else if (order check == False){ #check for order for rank > 1
                    Print NotImplementedError Wrong Order
                    return Null
                }
                collect the value from PyArrayObject

        Parameters:
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
        #TODO create and extern rank and order check function
        #check optional :
        if variable.is_optional :
            check = PyccelNot(VariableAddress(collect_var))
            body += [(check, [Assign(VariableAddress(variable), Nil())])]

        #rank check :
        check = PyccelNe(FunctionCall(numpy_get_ndims,[collect_var]), LiteralInteger(collect_var.rank))
        error = PyErr_SetString('PyExc_TypeError', '"{} must have rank {}"'.format(collect_var, str(collect_var.rank)))
        body  += [(check, [error, Return([Nil()])])]
        if check_type : #Type check
            numpy_dtype = self.find_in_numpy_dtype_registry(variable)
            arg_dtype   = self.find_in_dtype_registry(self._print(variable.dtype), variable.precision)
            check = PyccelNe(FunctionCall(numpy_get_type, [collect_var]), numpy_dtype)
            info_dump = PythonPrint([FunctionCall(numpy_get_type, [collect_var]), numpy_dtype])
            error = PyErr_SetString('PyExc_TypeError', '"{} must be {}"'.format(variable, arg_dtype))
            body += [(check, [info_dump, error, Return([Nil()])])]

        if collect_var.rank > 1 and self._target_language == 'fortran' :#Order check
            if collect_var.order == 'F':
                check = FunctionCall(numpy_check_flag,[collect_var, numpy_flag_f_contig])
            else:
                check = FunctionCall(numpy_check_flag,[collect_var, numpy_flag_c_contig])
                error = PyErr_SetString('PyExc_NotImplementedError',
                        '"Argument does not have the expected ordering ({})"'.format(collect_var.order))
                body += [(PyccelNot(check), [error, Return([Nil()])])]
        body += [(LiteralTrue(), [Assign(VariableAddress(variable),
                                self.get_collect_function_call(variable, collect_var))])]
        body = [If(*body)]

        return body

    def _body_management(self, used_names, variable, collect_var, cast_function, check_type = False):
        """
        Responsible for calling functions that take care of body creation
        """
        tmp_variable = None
        body = []

        if variable.rank > 0:
            body = self._body_array(variable, collect_var, check_type)

        elif variable.is_optional:
            tmp_variable = Variable(dtype=variable.dtype, name = self.get_new_name(used_names, variable.name+"_tmp"))
            body = self._body_optional_variable(tmp_variable, variable, collect_var, check_type)

        elif isinstance(variable, ValuedVariable):
            body = self._body_valued_variable(variable, collect_var, check_type)

        elif cast_function is not None:
            body = [Assign(variable, cast_function)]

        return body, tmp_variable

    # -------------------------------------------------------------------
    # Parsing arguments and building values Types functions
    # -------------------------------------------------------------------
    def get_PyArgParseType(self, used_names, variable):
        """
        Responsible for creating any necessary intermediate variables which are used
        to collect the result of PyArgParse, and collecting the required cast function

        Parameters:
        ----------
        used_names : list of strings
            List of variable and function names to avoid name collisions

        variable : Variable
            The variable which will be passed to the translated function

        Returns
        -------
        collect_var : Variable
            The variable which will be used to collect the argument

        cast_fun : FunctionCall
            call to cast function responsible of the conversion of one data type into another
        """
        cast_function = None
        collect_var = variable

        if variable.rank > 0:
            collect_type = PyccelPyArrayObject()
            collect_var = Variable(dtype= collect_type, is_pointer = True, rank = variable.rank,
                                    order= variable.order, name=self.get_new_name(used_names, variable.name+"_tmp"))

        elif isinstance(variable, ValuedVariable):
            collect_type = PyccelPyObject()
            collect_var = Variable(dtype=collect_type, is_pointer=True,
                name = self.get_new_name(used_names, variable.name+"_tmp"))

        elif variable.dtype is NativeBool():
            collect_type = NativeInteger()
            collect_var = Variable(dtype=collect_type, precision=4,
                name = self.get_new_name(used_names, variable.name+"_tmp"))
            cast_function =  self.get_cast_function_call('pyint_to_bool', collect_var)

        elif variable.dtype is NativeComplex():
            collect_type = PyccelPyObject()
            collect_var = Variable(dtype=collect_type, is_pointer=True,
                name = self.get_new_name(used_names, variable.name+"_tmp"))
            cast_function = self.get_cast_function_call('pycomplex_to_complex', collect_var)

        return collect_var, cast_function

    def get_PyBuildValue(self, used_names, variable):
        """
        Responsible for collecting the variable required to build the result
        and the necessary cast function

        Parameters:
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
            call to cast function responsible of the conversion of one data type into another
        """
        collect_var = variable
        cast_function = None

        if variable.dtype is NativeBool():
            collect_type = PyccelPyObject()
            collect_var = Variable(dtype=collect_type, is_pointer=True,
                name = self.get_new_name(used_names, variable.name+"_tmp"))
            cast_function = self.get_cast_function_call('bool_to_pyobj', variable)

        if variable.dtype is NativeComplex():
            collect_type = PyccelPyObject()
            collect_var = Variable(dtype=collect_type, is_pointer=True,
                name = self.get_new_name(used_names, variable.name+"_tmp"))
            cast_function = self.get_cast_function_call('complex_to_pycomplex', variable)
            self._to_free_PyObject_list.append(collect_var)

        return collect_var, cast_function

    #--------------------------------------------------------------------
    #                 _print_ClassName functions
    #--------------------------------------------------------------------

    def _print_IndexedElement(self, expr):
        assert(len(expr.indices)==1)
        return '{}[{}]'.format(self._print(expr.base.internal_variable), self._print(expr.indices[0]))

    def _print_PyccelPyObject(self, expr):
        return 'pyobject'

    def _print_PyccelPyArrayObject(self, expr):
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

    def _print_FunctionDef(self, expr):
        # Save all used names
        used_names = set([a.name for a in expr.arguments] + [r.name for r in expr.results] + [expr.name.name])

        # Find a name for the wrapper function
        wrapper_name = self.get_new_name(used_names.union(self._global_names), expr.name.name+"_wrapper")
        self._function_wrapper_names[expr.name] = wrapper_name
        self._global_names.add(wrapper_name)
        used_names.add(wrapper_name)
        # Collect local variables
        wrapper_vars        = {a.name : a for a in expr.arguments}
        wrapper_vars.update({r.name : r for r in expr.results})
        python_func_args    = self.get_new_PyObject("args"  , used_names)
        python_func_kwargs  = self.get_new_PyObject("kwargs", used_names)
        python_func_selfarg = self.get_new_PyObject("self"  , used_names)

        # Collect arguments and results
        wrapper_args    = [python_func_selfarg, python_func_args, python_func_kwargs]
        wrapper_results = [self.get_new_PyObject("result", used_names)]

        if any(isinstance(arg, FunctionAddress) for arg in expr.arguments):
            wrapper_func = FunctionDef(name = wrapper_name,
                arguments = wrapper_args,
                results = wrapper_results,
                body = [PyErr_SetString('PyExc_NotImplementedError', '"Cannot pass a function as an argument"'),
                        AliasAssign(wrapper_results[0], Nil()),
                        Return(wrapper_results)])
            return CCodePrinter._print_FunctionDef(self, wrapper_func)

        # Collect argument names for PyArgParse
        arg_names         = [a.name for a in expr.arguments]
        keyword_list_name = self.get_new_name(used_names,'kwlist')
        keyword_list      = PyArgKeywords(keyword_list_name, arg_names)

        wrapper_body              = [keyword_list]
        wrapper_body_translations = []

        parse_args = []
        collect_vars = {}
        for arg in expr.arguments:
            collect_var , cast_func = self.get_PyArgParseType(used_names, arg)
            collect_vars[arg] = collect_var

            body, tmp_variable = self._body_management(used_names, arg, collect_var, cast_func, True)
            if tmp_variable :
                wrapper_vars[tmp_variable.name] = tmp_variable

            # If the variable cannot be collected from PyArgParse directly
            wrapper_vars[collect_var.name] = collect_var

            # Save cast to argument variable
            wrapper_body_translations.extend(body)

            parse_args.append(collect_var)

            # Write default values
            if isinstance(arg, ValuedVariable):
                wrapper_body.append(self.get_default_assign(parse_args[-1], arg))

        # Parse arguments
        parse_node = PyArg_ParseTupleNode(python_func_args, python_func_kwargs, expr.arguments, parse_args, keyword_list)
        wrapper_body.append(If((PyccelNot(parse_node), [Return([Nil()])])))
        wrapper_body.extend(wrapper_body_translations)

        # Call function
        static_function, static_args, additional_body = self._get_static_function(used_names, expr, collect_vars)
        wrapper_body.extend(additional_body)
        for var in static_args :
            wrapper_vars[var.name] = var

        if len(expr.results)==0:
            func_call = FunctionCall(static_function, static_args)
        else:
            results   = expr.results if len(expr.results)>1 else expr.results[0]
            func_call = Assign(results,FunctionCall(static_function, static_args))

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
        self._to_free_PyObject_list.clear()
        #Return
        wrapper_body.append(Return(wrapper_results))

        # Create FunctionDef and write using classic method
        wrapper_func = FunctionDef(name = wrapper_name,
            arguments = wrapper_args,
            results = wrapper_results,
            body = wrapper_body,
            local_vars = wrapper_vars.values())
        return CCodePrinter._print_FunctionDef(self, wrapper_func)

    def _print_Module(self, expr):
        self._global_names = set(f.name.name for f in expr.funcs)
        self._module_name  = expr.name
        sep = self._print(SeparatorComment(40))
        if self._target_language == 'fortran':
            static_funcs = [as_static_function_call(f, expr.name, name=f.name) for f in expr.funcs]
        else:
            static_funcs = expr.funcs
        function_signatures = '\n'.join('{};'.format(self.function_signature(f)) for f in static_funcs)

        function_defs = '\n\n'.join(self._print(f) for f in expr.funcs)
        cast_functions = '\n\n'.join(CCodePrinter._print_FunctionDef(self, f)
                                        for f in self._cast_functions_dict.values())
        method_def_func = ',\n'.join(('{{\n'
                                     '"{name}",\n'
                                     '(PyCFunction){wrapper_name},\n'
                                     'METH_VARARGS | METH_KEYWORDS,\n'
                                     '"{doc_string}"\n'
                                     '}}').format(
                                            name = f.name,
                                            wrapper_name = self._function_wrapper_names[f.name],
                                            doc_string = f.doc_string)
                                     for f in expr.funcs)

        method_def_name = self.get_new_name(self._global_names, '{}_methods'.format(expr.name))
        method_def = ('static PyMethodDef {method_def_name}[] = {{\n'
                        '{method_def_func},\n'
                        '{{ NULL, NULL, 0, NULL}}\n'
                        '}};'.format(method_def_name = method_def_name ,method_def_func = method_def_func))

        module_def_name = self.get_new_name(self._global_names, '{}_module'.format(expr.name))
        module_def = ('static struct PyModuleDef {module_def_name} = {{\n'
                'PyModuleDef_HEAD_INIT,\n'
                '/* name of module */\n'
                '\"{mod_name}\",\n'
                '/* module documentation, may be NULL */\n'
                'NULL,\n' #TODO: Add documentation
                '/* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */\n'
                '-1,\n'
                '{method_def_name}\n'
                '}};'.format(module_def_name = module_def_name, mod_name = expr.name, method_def_name = method_def_name))

        init_func = ('PyMODINIT_FUNC PyInit_{mod_name}(void)\n{{\n'
                'PyObject *m;\n\n'
                'import_array();\n\n'
                'm = PyModule_Create(&{module_def_name});\n'
                'if (m == NULL) return NULL;\n\n'
                'return m;\n}}'.format(mod_name=expr.name, module_def_name = module_def_name))

        # Print imports last to be sure that all additional_imports have been collected
        imports  = [Import(s) for s in self._additional_imports]
        imports += [Import('Python')]
        imports += [Import('numpy/arrayobject')]
        imports  = '\n'.join(self._print(i) for i in imports)

        numpy_max_acceptable_version = [1, 19]
        numpy_current_version = [int(v) for v in np.version.version.split('.')[:2]]
        numpy_api_macro = '#define NPY_NO_DEPRECATED_API NPY_{}_{}_API_VERSION'.format(
                min(numpy_max_acceptable_version[0], numpy_current_version[0]),
                min(numpy_max_acceptable_version[1], numpy_current_version[1]))

        return ('#define PY_SSIZE_T_CLEAN\n'
                '{numpy_api_macro}\n'
                '{imports}\n\n'
                '{function_signatures}\n\n'
                '{sep}\n\n'
                '{cast_functions}\n\n'
                '{sep}\n\n'
                '{function_defs}\n\n'
                '{method_def}\n\n'
                '{sep}\n\n'
                '{module_def}\n\n'
                '{sep}\n\n'
                '{init_func}\n'.format(
                    numpy_api_macro = numpy_api_macro,
                    imports = imports,
                    function_signatures = function_signatures,
                    sep = sep,
                    cast_functions = cast_functions,
                    function_defs = function_defs,
                    method_def = method_def,
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

    return CWrapperCodePrinter(parser, target_language, settings).doprint(expr, assign_to)
