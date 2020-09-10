# coding: utf-8
# pylint: disable=R0201

from pyccel.codegen.printing.ccode import CCodePrinter

from pyccel.ast.builtins import Bool

from pyccel.ast.core import Variable, ValuedVariable, Assign, AliasAssign, FunctionDef
from pyccel.ast.core import If, Nil, Return, FunctionCall, PyccelNot
from pyccel.ast.core import create_incremented_string, Declare

from pyccel.ast.datatypes import NativeInteger, NativeBool

from pyccel.ast.cwrapper import PyccelPyObject, PyArg_ParseTupleNode, PyBuildValueNode
from pyccel.ast.cwrapper import PyArgKeywords, CastFunction

from pyccel.ast.type_inference import str_dtype

from pyccel.errors.errors import Errors
from pyccel.errors.messages import *

errors = Errors()

__all__ = ["CWrapperCodePrinter", "cwrappercode"]

class CWrapperCodePrinter(CCodePrinter):
    def __init__(self, parser, settings={}):
        CCodePrinter.__init__(self, parser,settings)
        self._cast_functions_set = set()

    def get_new_name(self, used_names, requested_name):
        if requested_name not in used_names:
            used_names.add(requested_name)
            return requested_name
        else:
            incremented_name, counter = create_incremented_string(used_names, prefix=requested_name)
            return incremented_name

    def pop_cast_function(self, cast_func, cast_functions_list):
        if cast_func in cast_functions_list:
            for i in cast_functions_list:
                if i == cast_func:
                    return i
        return cast_func

    
    def get_cast_function(self, used_names, cast_type, from_variable, to_variable):
        cast_function_arg = [from_variable]
        cast_function_result = [to_variable]
        cast_function_body = [Return(cast_function_result)]
        cast_function_name = self.get_new_name(used_names, cast_type)
        cast_function = CastFunction(cast_function_name, cast_type, 
                            cast_function_arg, cast_function_body, cast_function_result)
        return cast_function

    def get_PyArgParseType(self, used_names, variable):
        if variable.dtype is NativeBool():
            collect_type = NativeInteger()
            collect_var = Variable(dtype=collect_type, precision=4, 
                name = self.get_new_name(used_names, variable.name+"_tmp"))
            cast_function = self.get_cast_function(used_names, 'pyint_to_bool', collect_var, variable)
            return collect_var , cast_function
        return variable, None

    def get_PyBuildeValue(self, used_names, variable):
        if variable.dtype is NativeBool():
            collect_type = NativeInteger()
            collect_var = Variable(dtype=collect_type, precision=4, 
            name = self.get_new_name(used_names, variable.name+"_tmp"))
            cast_function = self.get_cast_function(used_names, 'pyint_to_bool', collect_var, variable)
            return collect_var , cast_function
        return variable, None

    def _print_PyccelPyObject(self, expr):
        return 'pyobject'

    def _print_PyArg_ParseTupleNode(self, expr):
        name = 'PyArg_ParseTupleAndKeywords'
        pyarg = expr.pyarg
        pykwarg = expr.pykwarg
        flags = expr.flags
        args = ','.join(['&{}'.format(self._print(a)) for a in expr.args])
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
        name = 'Py_BuildValue'
        flags = expr.flags
        args = ','.join(['{}'.format(self._print(a)) for a in expr.args])        
        #to change for args rank 1 +
        if expr.args:
            code = '{name}("{flags}", {args})'.format(name=name, flags=flags, args=args)
        else :
            code = '{name}("")'.format(name=name)
        return code

    def _print_PyArgKeywords(self, expr):
        arg_names = ', \n'.join(['"{}"'.format(a) for a in expr.arg_names] + [self._print(Nil())])
        return ('static char *{name}[] = {{\n'
                        '{arg_names}\n'
                        '}};\n'.format(name=expr.name, arg_names = arg_names))

    def _print_CastFunction(self, expr):
        decs = [Declare(i.dtype, i) for i in expr.results]
        decs       = '\n'.join(self._print(i) for i in decs)
        body = self._print(expr.body[0])
        return '{0}\n{{\n{1}\n{2}\n}}\n'.format(self.function_signature(expr), decs, body)

    def _print_FunctionDef(self, expr):
        used_names = set([a.name for a in expr.arguments] + [r.name for r in expr.results] + [expr.name.name])
        wrapper_vars = [a for a in expr.arguments] + [r for r in expr.results]
        python_func_args = Variable(dtype=PyccelPyObject(),
                                 name=self.get_new_name(used_names, "args"),
                                 is_pointer=True)
        python_func_kwargs = Variable(dtype=PyccelPyObject(),
                                 name=self.get_new_name(used_names, "kwargs"),
                                 is_pointer=True)
        wrapper_args = [Variable(dtype=PyccelPyObject(),
                                 name=self.get_new_name(used_names, "self"),
                                 is_pointer=True),
                        python_func_args, python_func_kwargs]
        wrapper_results = [Variable(dtype=PyccelPyObject(),
                                    name=self.get_new_name(used_names, "result"),
                                    is_pointer=True)]

        arg_names = [a.name for a in expr.arguments]
        keyword_list_name = self.get_new_name(used_names,'kwlist')
        keyword_list = PyArgKeywords(keyword_list_name, arg_names)
        cast_functions_list = set()

        wrapper_body = [keyword_list]
        wrapper_body_translations = []

        parse_args = []
        # TODO: Simplify (to 1 line?)
        # TODO: Handle optional args
        for a in expr.arguments:
            collect_var, cast_func = self.get_PyArgParseType(used_names, a)
            if cast_func is not None:
                # TODO: Add other properties
                cast_func = self.pop_cast_function(cast_func, cast_functions_list)
                wrapper_vars.append(collect_var)
                parse_args.append(collect_var)
                cast_func_call = FunctionCall(cast_func, [collect_var])
                wrapper_body_translations.append(AliasAssign(a, cast_func_call))
                cast_functions_list.add(cast_func) 
            else:
                parse_args.append(a)
            # TODO: Handle assignment to PyObject for default variables
            if isinstance(a, ValuedVariable):
                wrapper_body.append(Assign(parse_args[-1],a.value))

        parse_node = PyArg_ParseTupleNode(python_func_args, python_func_kwargs, expr.arguments, parse_args, keyword_list)
        wrapper_body.append(If((PyccelNot(parse_node), [Return([Nil()])])))
        wrapper_body.extend(wrapper_body_translations)


        if len(expr.results)==0:
            func_call = FunctionCall(expr, expr.arguments)
        else:
            results = expr.results if len(expr.results)>1 else expr.results[0]
            func_call = AliasAssign(results,FunctionCall(expr, expr.arguments))
        wrapper_body.append(func_call)
        #TODO: Loop over results to carry out necessary casts and collect Py_BuildValue type string
        res_args = []
        for a in expr.results :
            collect_var, cast_func = self.get_PyBuildeValue(used_names, a)
            if cast_func is not None :
                cast_func = self.pop_cast_function(cast_func, cast_functions_list)                
                wrapper_vars.append(collect_var)
                res_args.append(collect_var)
                cast_func_call = FunctionCall(cast_func, [a])
                wrapper_body.append(AliasAssign(collect_var, cast_func_call))
                cast_functions_list.add(cast_func) 
            else :
                res_args.append(a)

        code = '\n'.join(self._print(i) for i in cast_functions_list if i not in self._cast_functions_set)
        self._cast_functions_set.update(cast_functions_list)
        wrapper_body.append(AliasAssign(wrapper_results[0],PyBuildValueNode(res_args)))
        wrapper_body.append(Return(wrapper_results))

        wrapper_name = self.get_new_name(used_names, expr.name.name+"_wrapper")
        #TODO: Create node and add args
        wrapper_func = FunctionDef(name = wrapper_name,
            arguments = wrapper_args,
            results = wrapper_results,
            body = wrapper_body,
            local_vars = wrapper_vars)
        code += CCodePrinter._print_FunctionDef(self, wrapper_func)
        return code

    def _print_Module(self, expr):
        function_signatures = '\n'.join('{};'.format(self.function_signature(f)) for f in expr.funcs)

        function_defs = '\n'.join(self._print(f) for f in expr.funcs)

        methode_def_func = ',\n'.join("    {{ \"{0}\", (PyCFunction){0}_wrapper, METH_VARARGS | METH_KEYWORDS, \"{1}\" }}".format(
            f.name,f.doc_string) for f in expr.funcs)
        
        methode_def = ('static PyMethodDef {mod_name}_methods[] = {{\n'
                        '{methode_def_func}'
                        ',\n    {{ NULL, NULL, 0, NULL}}'
                        '\n}};\n\n'.format(mod_name = expr.name ,methode_def_func = methode_def_func))
        
        module_def = ('static struct PyModuleDef {mod_name}_module = {{\n'
                '   PyModuleDef_HEAD_INIT,\n'
                '   \"{mod_name}\",   /* name of module */\n'
                '   NULL, /* module documentation, may be NULL */\n'
                '   -1,       /* size of per-interpreter state of the module,\n'
                '                 or -1 if the module keeps state in global variables. */\n'
                '   {mod_name}_methods\n'
                '}};\n\n'.format(mod_name = expr.name))

        init_func = ('PyMODINIT_FUNC PyInit_{mod_name}(void)\n{{\n'
                'PyObject *m;\n\n'
                'm = PyModule_Create(&{mod_name}_module);\n'
                'if (m == NULL) return NULL;\n\n'
                'return m;\n}}'.format(mod_name=expr.name))

        return ('#define PY_SSIZE_T_CLEAN\n'
                '#include <Python.h>\n\n'
                '{function_signatures}\n\n'
                '{function_defs}\n\n'
                '{methode_def}\n'
                '{module_def}\n\n'
                '{init_func}\n'.format(
                    function_signatures = function_signatures,
                    function_defs = function_defs,
                    methode_def = methode_def,
                    module_def = module_def,
                    init_func = init_func))

def cwrappercode(expr, parser, assign_to=None, **settings):
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

    return CWrapperCodePrinter(parser, settings).doprint(expr, assign_to)
