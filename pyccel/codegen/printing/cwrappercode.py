from pyccel.codegen.printing.ccode import CCodePrinter
from pyccel.ast.core import Variable
from pyccel.ast.datatypes import NativeInteger, NativeReal, NativeComplex, NativeBool, NativeString
from pyccel.ast.cwrapper import PyccelPyObject, PyArg_ParseTupleNode, PyBuildValueNode

from pyccel.errors.errors import Errors
from pyccel.errors.messages import *

errors = Errors()

__all__ = ["CWrapperCodePrinter", "ccode"]

pytype_registry = {
        NativeInteger(): 'l',
        NativeReal(): 'd',
        NativeComplex():'c',
        NativeBool():'p',
        NativeString():'s',
        PyccelPyObject():'O'
        }

class CWrapperCodePrinter(CCodePrinter):
    def __init__(self, parser, settings={}):
        CCodePrinter.__init__(self, parser,settings)

    def get_new_name(self, used_names, requested_name):
        if requested_name not in used_names:
            used_names.add(requested_name)
            return requested_name
        else:
            return create_incremented_string(used_names, prefix=requested_name)

    def get_PyArgParseType(self, dtype):
        #TODO: Depends on type, rank, etc
        if dtype is NativeBool():
            return NativeInteger(), lambda arg, tmp: Assign(arg, Bool(tmp))
        return dtype, None

    def _print_FunctionDef(self, expr):
        used_names = set([a.name for a in expr.arguments] + [r.name for r in expr.results])
        wrapper_vars = [a for a in expr.arguments] + [r for r in expr.results]
        python_func_args = Variable(dtype=PyccelPyObject(),
                                 name=get_new_name(used_names, "args"),
                                 is_pointer=True)
        wrapper_args = [Variable(dtype=PyccelPyObject(),
                                 name=get_new_name(used_names, "self"),
                                 is_pointer=True),
                        python_func_args]
        wrapper_results = [Variable(dtype=PyccelPyObject(),
                                    name=get_new_name(used_names, "result"),
                                    is_pointer=True)]
        wrapper_body = []
        parse_args = []
        type_keys = ''
        for a in expr.arguments:
            collect_type, cast_func = self.get_PyArgParseType(a.dtype)
            if cast_func is not None:
                # TODO: Add other properties
                collect_var = Variable(dtype=collect_type,
                        name=get_new_name(used_names, a.name+"_tmp"))
                wrapper_vars.append(collect_var)
                parse_args.append(collect_var)
                wrapper_body.append(cast_func(a, collect_var))
            else:
                parse_args.append(a)
            type_keys.append(pytype_registry[str_dtype(arg.dtype)])

        # TODO: Create PyArg_ParseTupleNode
        wrapper_body.insert(0,If((Not(PyArg_ParseTupleNode(python_func_args, type_keys, parse_args)), Return(Nil))))

        if len(expr.results)==0:
            func_call = FunctionCall(expr, expr.arguments)
        else:
            results = expr.results if len(expr.results)>1 else expr.results[0]
            func_call = Assign(results,FunctionCall(expr, expr.arguments))
        wrapper_body.append(func_call)

        #TODO: Loop over results to carry out necessary casts and collect Py_BuildValue type string

        #TODO: Create node and add args
        wrapper_body.append(Return(PyBuildValueNode()))

        wrapper_func = FunctionDef(name = self.parser.get_new_name(expr.name+"_wrapper"),
            arguments = wrapper_args,
            results = wrapper_results,
            body = wrapper_body,
            local_vars = wrapper_vars)

        return CWrapper._print_FunctionDef(wrapper_func)

    def _print_Module(self, expr):
        function_signatures = '\n'.join('{};'.format(self.function_signature(f)) for f in expr.funcs)

        function_defs = '\n'.join(self._print(f) for f in expr.funcs)

        #TODO: Print ModuleDef (see cwrapper.py L69)

        init_func = ('PyMODINIT_FUNC PyInit_{mod_name}(void)\n{{\n'
                'PyObject *m;\n\n'
                'm = PyModule_Create(&{mod_name});\n'
                'if (m == NULL) return NULL;\n\n'
                'return m;\n}}'.format(mod_name=expr.mod_name))

        return ('#define PY_SSIZE_T_CLEAN\n'
                '#include <Python.h>\n'
                '{function_signatures}\n'
                '{function_defs}\n'
                '{module_def}\n'
                '{init_func}\n'.format(
                    function_signatures = function_signatures,
                    function_defs = function_defs,
                    module_def = module_def,
                    init_func = init_func))

def cwrappercode(expr, assign_to=None, **settings):
    """Converts an expr to a string of c wrapper code

    expr : Expr
        A sympy expression to be converted.
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

    return CWrapperCodePrinter(settings).doprint(expr, assign_to)
