# coding: utf-8
# pylint: disable=R0201
from collections import OrderedDict

from pyccel.codegen.printing.ccode import CCodePrinter

from pyccel.ast.numbers   import BooleanTrue
from pyccel.ast.builtins import Bool

from pyccel.ast.core import Variable, ValuedVariable, Assign, AliasAssign, FunctionDef
from pyccel.ast.core import If, Nil, Return, FunctionCall, PyccelNot, Symbol
from pyccel.ast.core import create_incremented_string, Declare, SeparatorComment

from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeComplex

from pyccel.ast.cwrapper import PyccelPyObject, PyArg_ParseTupleNode, PyBuildValueNode
from pyccel.ast.cwrapper import PyArgKeywords, PyccelPyComplex, FuncCall

from pyccel.ast.type_inference import str_dtype

from pyccel.errors.errors import Errors
from pyccel.errors.messages import *

errors = Errors()

__all__ = ["CWrapperCodePrinter", "cwrappercode"]

dtype_registry = {('pyobject', 0) : 'PyObject',
                  ('pycomplex', 0) : 'Py_complex'}

class CWrapperCodePrinter(CCodePrinter):
    def __init__(self, parser, settings={}):
        CCodePrinter.__init__(self, parser,settings)
        self._cast_functions_dict = OrderedDict()
        self._function_wrapper_names = dict()
        self._global_names = set()

    def get_new_name(self, used_names, requested_name):
        if requested_name not in used_names:
            used_names.add(requested_name)
            return requested_name
        else:
            incremented_name, _ = create_incremented_string(used_names, prefix=requested_name)
            return incremented_name

    def find_in_dtype_registry(self, dtype, prec):
        try :
            return dtype_registry[(dtype, prec)]
        except KeyError:
            return CCodePrinter.find_in_dtype_registry(self, dtype, prec)

    def get_cast_function(self, used_names, cast_type, from_variable, to_variable):
        if cast_type in self._cast_functions_dict:
            return self._cast_functions_dict[cast_type]

        cast_function_name = self.get_new_name(used_names, cast_type)
        cast_function_arg = [from_variable]
        cast_function_result = [to_variable]
        cast_function_local_vars = []
 
        #switch case of cast_type
        if cast_type == 'pyint_to_bool':
            cast_function_body = [Assign(cast_function_result[0], Bool(cast_function_arg[0]))]
        elif cast_type == 'bool_to_pyobj':
            cast_function_body = [If((Bool(cast_function_arg[0]),
                [Assign(cast_function_result[0], Symbol('Py_True'))]),
                (BooleanTrue(), [Assign(cast_function_result[0], Symbol('Py_False'))]))]
        elif cast_type == 'pycomplex_to_complex':
            cast_function_body = [Assign(cast_function_result[0],
                FuncCall('__builtin_complex', [FuncCall('PyComplex_RealAsDouble', cast_function_arg),
                                               FuncCall('PyComplex_ImagAsDouble', cast_function_arg)]))]            
        elif cast_type == 'complex_to_pycomplex':
            cast_function_body = [Assign(cast_function_result[0],
                FuncCall('PyComplex_FromDoubles', [FuncCall('__real', cast_function_arg),
                                                   FuncCall('__imag', cast_function_arg)]))] 
        cast_function_body += [Return(cast_function_result)]
        cast_function = FunctionDef(name = cast_function_name,
                                    arguments = cast_function_arg,
                                    body = cast_function_body,
                                    results = cast_function_result,
                                    local_vars = cast_function_local_vars)
        self._cast_functions_dict[cast_type] = cast_function
        return cast_function

    def get_PyArgParseType(self, used_names, variable):
        if variable.dtype is NativeBool():
            collect_type = NativeInteger()
            collect_var = Variable(dtype=collect_type, precision=4,
                name = self.get_new_name(used_names, variable.name+"_tmp"))
            cast_function = self.get_cast_function(used_names, 'pyint_to_bool', variable, collect_var)
            return collect_var , cast_function
        if variable.dtype is NativeComplex():
            collect_type = PyccelPyObject()
            collect_var = Variable(dtype=collect_type, rank = 1,
                name = self.get_new_name(used_names, variable.name+"_tmp"))             
            cast_function = self.get_cast_function(used_names, 'pycomplex_to_complex', collect_var, variable)
            return collect_var , cast_function
        return variable, None

    def get_PyBuildValue(self, used_names, variable):
        if variable.dtype is NativeBool():
            collect_type = PyccelPyObject()
            collect_var = Variable(dtype=collect_type, rank = 1,
                name = self.get_new_name(used_names, variable.name+"_tmp"))
            cast_function = self.get_cast_function(used_names, 'bool_to_pyobj', variable, collect_var)
            return collect_var , cast_function
        if variable.dtype is NativeComplex():
            collect_type = PyccelPyObject()
            collect_var = Variable(dtype=collect_type, rank = 1,
                name = self.get_new_name(used_names, variable.name+"_tmp"))
            cast_function = self.get_cast_function(used_names, 'complex_to_pycomplex', variable, collect_var)        
            return collect_var , cast_function
        return variable, None

    def _print_PyccelPyObject(self, expr):
        return 'pyobject'

    def _print_PyccelPyComplex(self, expr):
        return 'pycomplex'

    def _print_FuncCall(self, expr):
        name = expr.name
        args = ','.join(['{}'.format(self._print(a)) for a in expr.args])
        return ('{name}({args})'.format(
            name = name,
            args = args))

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

    def _print_FunctionDef(self, expr):
        # Save all used names
        used_names = set([a.name for a in expr.arguments] + [r.name for r in expr.results] + [expr.name.name])

        # Find a name for the wrapper function
        wrapper_name = self.get_new_name(used_names.union(self._global_names), expr.name.name+"_wrapper")
        self._function_wrapper_names[expr.name] = wrapper_name
        self._global_names.add(wrapper_name)
        used_names.add(wrapper_name)

        # Collect local variables
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

        wrapper_body = [keyword_list]
        wrapper_body_translations = []

        parse_args = []
        # TODO: Simplify (to 1 line?)
        # TODO (After PR 422): Handle optional args
        for a in expr.arguments:
            collect_var, cast_func = self.get_PyArgParseType(used_names, a)
            if cast_func is not None:
                wrapper_vars.append(collect_var)
                cast_func_call = FunctionCall(cast_func, [collect_var])
                wrapper_body_translations.append(Assign(a, cast_func_call))

            parse_args.append(collect_var)

            if isinstance(a, ValuedVariable):
                wrapper_body.append(Assign(parse_args[-1],a.value))
        
        parse_node = PyArg_ParseTupleNode(python_func_args, python_func_kwargs, expr.arguments, parse_args, keyword_list)
        wrapper_body.append(If((PyccelNot(parse_node), [Return([Nil()])])))
        wrapper_body.extend(wrapper_body_translations)

        if len(expr.results)==0:
            func_call = FunctionCall(expr, expr.arguments)
        else:
            results = expr.results if len(expr.results)>1 else expr.results[0]
            func_call = Assign(results,FunctionCall(expr, expr.arguments))
        wrapper_body.append(func_call)
        #TODO: Loop over results to carry out necessary casts and collect Py_BuildValue type string
        res_args = []
        for a in expr.results :
            collect_var, cast_func = self.get_PyBuildValue(used_names, a)
            if cast_func is not None:
                wrapper_vars.append(collect_var)
                cast_func_call = FunctionCall(cast_func, [a])
                wrapper_body.append(Assign(collect_var, cast_func_call))

            res_args.append(collect_var)
        wrapper_body.append(AliasAssign(wrapper_results[0],PyBuildValueNode(res_args)))
        wrapper_body.append(Return(wrapper_results))
        #TODO: Create node and add args
        wrapper_func = FunctionDef(name = wrapper_name,
            arguments = wrapper_args,
            results = wrapper_results,
            body = wrapper_body,
            local_vars = wrapper_vars)
        return CCodePrinter._print_FunctionDef(self, wrapper_func)

    def _print_Module(self, expr):
        self._global_names = set([f.name.name for f in expr.funcs])
        sep = self._print(SeparatorComment(40))
        function_signatures = '\n'.join('{};'.format(self.function_signature(f)) for f in expr.funcs)

        function_defs = '\n\n'.join(self._print(f) for f in expr.funcs)
        cast_functions = '\n\n'.join(CCodePrinter._print_FunctionDef(self, f) for f in self._cast_functions_dict.values())
        method_def_func = ',\n'.join("{{ \"{name}\", (PyCFunction){wrapper_name}, METH_VARARGS | METH_KEYWORDS, \"{doc_string}\" }}".format(
            name = f.name,
            wrapper_name = self._function_wrapper_names[f.name],
            doc_string = f.doc_string) for f in expr.funcs)

        method_def_name = self.get_new_name(self._global_names, '{}_methods'.format(expr.name))
        method_def = ('static PyMethodDef {method_def_name}[] = {{\n'
                        '{method_def_func}'
                        ',\n    {{ NULL, NULL, 0, NULL}}'
                        '\n}};'.format(method_def_name = method_def_name ,method_def_func = method_def_func))

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
                'm = PyModule_Create(&{module_def_name});\n'
                'if (m == NULL) return NULL;\n\n'
                'return m;\n}}'.format(mod_name=expr.name, module_def_name = module_def_name))

        return ('#define PY_SSIZE_T_CLEAN\n'
                '#include <Python.h>\n\n'
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
                    function_signatures = function_signatures,
                    sep = sep,
                    cast_functions = cast_functions,
                    function_defs = function_defs,
                    method_def = method_def,
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
