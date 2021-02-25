Plan:
#TODO using converter function optional variable need to be treated as ** (double pointer) in coverter functions
helpers ():
c_binding():

generate_convert_array():

generate_convert_scalar():

parseargs()
buildvalue()

print_Interface():

Print_FunctionDef():
    --> generate new wrapper name
    --> collect used_names #it can be droped
    --> collect wrapper variable
    --> collect key_words
    --> loop on all function arguments :
        --> generate convert function
        --> generate new_args (needed c binding) #it can be droped

    --> generate PyArg_ParseNode
    --> generate static functioncall

    --> loop on all  function results :
        --> generate convert function or use the old principe

    --> build the FunctionDef
    --> print the FunctionDef

Print_Module():

from pyccel.ast.core        import create_incremented_string, SeparatorComment

from pyccel.ast.core        import FunctionCall, FunctionDef
from pyccel.ast.core        import Assign, AliasAssign, Nil
from pyccel.ast.core        import If, IfSection, PyccelEq

from pyccel.ast.cwrapper    import (PyArgKeywordsm, PyArg_ParseTupleNode,
                                    PyBuildValueNode)

from pyccel.ast.cwrapper    import PyccelPyObject, PyNone
from pyccel.ast.cwrapper    import get_custom_key
from pyccel.ast.variable    import Variable, ValuedVariable, VariableAddress

from pyccel.ast.bind_c      import as_static_function_call

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

    def get_new_name(self, used_names, requested_name):
        """
        """
        if requested_name not in used_names:
            used_names.add(requested_name)
            return requested_name
        else:
            incremented_name, _ = create_incremented_string(used_names, prefix=requested_name)
            return incremented_name

    def get_wrapper_name(self, used_names, function):
        """
        """
        name = function.name
        wrapper_name = self.get_new_name(used_names.union(self._global_names), name+"_wrapper")

        self._function_wrapper_names[func.name] = wrapper_name
        self._global_names.add(wrapper_name)
        used_names.add(wrapper_name)

        return wrapper_name

    def get_new_PyObject(self, name, used_names):
        """
        """
        return Variable(dtype      = PyccelPyObject(),
                        name       = self.get_new_name(used_names, name),
                        is_pointer = True)


    def get_wrapper_arguments(self, used_names):
        """
        """
        python_func_args    = self.get_new_PyObject("args"  , used_names)
        python_func_kwargs  = self.get_new_PyObject("kwargs", used_names)
        python_func_selfarg = self.get_new_PyObject("self"  , used_names)

        return [python_func_selfarg, python_func_args, python_func_kwargs]


    def get_static_function(self, function):
        """
        """
        if self._target_language == 'fortran':
            static_func = as_static_function_call(function,
                                                  self._module_name,
                                                  name = function.name)
        else:
            static_func = function

        return static_func

    def generate_valued_variable_body(self, py_variable, c_variable):
        """
        """
        check = PyccelEq(VariableAddress(py_variable), VariableAddress(Py_None))
        if c_variable.is_optional:
            body =  [AliasAssign(variable, Nil())]

        else:
            body = [Assign(variable, variable.value)]

        return [IfSection(check, body)]

    #--------------------------------------------------------------------
    #                   Convert functions
    #--------------------------------------------------------------------

    def generate_scalar_converter_function(self, used_names, variable, check_is_needed = True):
        """
        """

        func_name       = 'py_to_{}'.format(self._print(variable.dtype))
        py_variable     = self.get_new_PyObject('o', used_names)
        c_variable      = Variable.clone(name = 'c', is_pointer = True)
        body            = []

        if isinstance(variable, ValuedVariable):
            body.append(generate_valued_variable_body(py_object, variable))

        body.append(generate_numpy_type_body()) #TODO
        body.append(generate_python_type_body()) #TODO

        if check_is_needed:
            body.append(IfSection(LiteralTrue(), [generate_type_error()])) #TODO

        body    = [If(*body)]
        funcDef = FunctionDef(name     = func_name,
                            arguments  = [py_variable, c_variable],
                            results    = [],
                            body       = body)

        return funcDef

    def generate_array_converter_function(self, used_names, variable, check_is_needed = True):
        """
        """

        func_name       = 'py_to_{}'.format(self._print(variable.dtype))
        py_variable     = self.get_new_PyObject('o', used_names)
        c_variable      = Variable.clone(name = 'c', is_pointer = True)
        body            = []

        body.append(IfSection(PyArray_CheckRank(), [generate_rank_error()])) #TODO
        body.append(IfSection(PyArray_CheckOrder(),[generate_order_error()])) #TODO

        if check_is_needed:
            body.append(IfSection(PyArray_CheckType, [generate_type_error()])) #TODO

        body.append(IfSection(LiteralTrue(), [PyArray_to_Array()])) #TODO

        body    = [If(*body)]
        funcDef = FunctionDef(name     = func_name,
                            arguments  = func_arguments,
                            results    = [],
                            body       = fbody)

        return funcDef

    def generate_tuple_converter_function(self, used_names, variable):
        #TODO

    def generate_pyobject_converter_function(self, used_names, variable):
        """
        """

        func_name       = '{}_to_py'.format(self._print(variable.dtype))

        func_arguments  = [self.get_new_PyObject('o', used_names)]
        func_arguments += [variable.clone(name = self.get_new_name(used_name, variable.name),
                                is_pointer = True)]
        local_vars      = []
        func_body       = #TODO]

        funcDef = FunctionDef(name     = func_name,
                            arguments  = func_arguments,
                            results    = [],
                            local_vars = local_vars,
                            body       = func_body)

        return funcDef


    # -------------------------------------------------------------------
    #       Parsing arguments and building values  functions
    # -------------------------------------------------------------------

    def get_PyArgParse_Converter_Function(self, variable):
        """
        """
        if get_custom_key(variable) in self.converter_functions_dict:

            if variable.rank > 0:
                function = self.generate_array_converter_function(variable)
            else:
                function = self.generate_scalar_converter_function(variable)

            self.converter_functions_dict[get_custom_key(variable)]

    def get_PyBuildValue_Converter_function(self, variable):
        """
        """
        if variable.rank > 0:
            raise NotImplementedError(
            'return not implemented for this datatype : {}'.format(variable.dtype))

        try:
            python_to_c_registry[(variable.dtype, variable.precision)]
        except: KeyError
            raise NotImplementedError(
            'return not implemented for this datatype : {}'.format(variable.dtype))

    #--------------------------------------------------------------------
    #                 _print_ClassName functions
    #--------------------------------------------------------------------

    def _print_PyccelPyObject(self, expr):
        return 'pyobject'

    def _print_PyArgKeywords(self, expr):
        arg_names  = ['"{}"'.format(a) for a in expr.arg_names]
        arg_names += [self._print(Nil())]
        
        arg_names  = ',\n'.join(arg_names)
        code       = 'static char *{name}[] = {{\n{arg_names}\n}};\n'

        return  code.format(name = expr.name, arg_names = arg_names))

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
        # TODO nightmare

    def _print_FunctionDef(self, expr):
        # Save all used names
        used_names = set([a.name for a in expr.arguments]
                    + [r.name for r in expr.results]
                    + [expr.name])

        # Find a name for the wrapper function
        wrapper_name = self.get_wrapper_name(used_names, expr)

        # Collect arguments and results
        wrapper_args    = get_wrapper_arguments(used_names)
        wrapper_results = [self.get_new_PyObject("result", used_names)]

        arg_names         = [a.name for a in expr.arguments]
        keyword_list_name = self.get_new_name(used_names, 'kwlist')
        keyword_list      = PyArgKeywords(keyword_list_name, arg_names)

        wrapper_body      = [keyword_list]
        func_args         = []
        
        for arg in expr.arguments:
            self.get_PyArgParse_Converter_Function(arg)
            func_args.append(None) #TODO Bind_C_Arg

        parse_node = PyArg_ParseTupleNode(*wrapper_args[:-1]
                                          self.converter_functions_dict,
                                          expr.arguments, keyword_list)

        wrapper_body.append(If(IfSection(PyccelNot(parse_node), [Return([Nil()])])))

        static_function = self.get_static_function

        function_call   = FunctionCall(static_function, func_args)
        
        if len(expr.results) > 0:
            results       = expr.results if len(expr.results)>1 else expr.results[0]
            function_call = Assign(results, function_call)
        
        wrapper_body.append(function_call)

        for res in expr.results:
            self.get_PyBuildValue_Converter_function(res)

        build_node = PyBuildValueNode(expr.results, self.building_converter_functions)

        wrapper_body.append(AliasAssign(wrapper_results[0], build_node))

        wrapper_function = FunctionDef(name     = wrapper_name,
                                    arguments   = wrapper_args,
                                    results     = wrapper_results,
                                    body        = wrapper_body,
                                    local_varts = tuple(func_args + expr.results))
        
        return CCodePrinter._print_FunctionDef(self, wrapper_func)

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
        imports += [Import('Python'), Import('cwrapper')]
        imports += [Import('numpy/arrayobject')]
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