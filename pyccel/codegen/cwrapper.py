from pyccel.ast.core import FunctionCall

from pyccel.codegen.printing.ccode import CCodePrinter, dtype_registry
from pyccel.ast.core import Module, Declare, Assign
from pyccel.ast.type_inference import str_dtype

pytype_registry = {
        'integer': 'l',
        'real': 'd',
        'complex':'c',
        'bool':'p',
        'str':'s'
        }

def write_python_wrapper(expr, printer):
    code  = "static PyObject * "+str(expr.name)+"_wrapper(PyObject *self, PyObject *args)\n{\n    "

    arg_decs = [Declare(i.dtype, i) for i in expr.arguments]
    arg_decs = '\n    '.join(printer._print(i) for i in arg_decs)

    results_decs = [Declare(i.dtype, i) for i in expr.results]
    results_decs = '\n    '.join(printer._print(i) for i in results_decs)
    code += '{0}\n    {1}\n    '.format(arg_decs, results_decs)

    # Check if the function has No arguments
    if not expr.arguments:
        code += "if (!PyArg_ParseTuple(args, \"\"))\n        return NULL;\n    "
    else:
        code += "if (!PyArg_ParseTuple(args, \""
        code += ''.join(pytype_registry[str_dtype(arg.dtype)] for arg in expr.arguments)
        code += "\", "
        code += ', '.join("&" + printer._print(arg) for arg in expr.arguments)
        code += "))\n        return NULL;\n    "

    if len(expr.results)==0:
        func_call = FunctionCall(expr, expr.arguments)
    else:
        results = expr.results if len(expr.results)>1 else expr.results[0]
        func_call = Assign(results,FunctionCall(expr, expr.arguments))
    code += printer._print(func_call)
    code += '\n'

    results_dtypes = ''.join(pytype_registry[str_dtype(arg.dtype)] for arg in expr.results)
    result_names = ', '.join(res.name for res in expr.results)
    code += "    return Py_BuildValue("
    if not expr.results: # case of function with no return value
        code += "\"\""
    else: # function with return value
        code += "\"{0}\", {1}".format(results_dtypes,result_names)
    code += ");\n"
    code += "}\n"
    return code

def create_c_wrapper(mod_name, codegen):
    assert(codegen.is_module)
    printer = CCodePrinter(codegen.parser)
    funcs = codegen.routines
    code  = """#define PY_SSIZE_T_CLEAN\n"""
    code += """#include <Python.h>\n\n"""
    code += '\n'.join("{0};".format(printer.function_signature(f)) for f in funcs) + '\n'
    code += '\n'
    code += '\n'.join([write_python_wrapper(f, printer) for f in funcs])
    code += '\n'
    code += "static PyMethodDef " + mod_name + "_methods[] = {\n"
    code += ',\n'.join("    {{ \"{0}\", {0}_wrapper, METH_VARARGS, \"{1}\" }}".format(f.name,f.doc_string) for f in funcs)
    code += ",\n    { NULL, NULL, 0, NULL }"
    code += "\n};\n\n"

    code += "static struct PyModuleDef " + mod_name + "_module = {\n"
    code += "    PyModuleDef_HEAD_INIT,\n"
    code += "    \"" + mod_name + "\",   /* name of module */\n"
    code += "    NULL, /* module documentation, may be NULL */\n"
    code += "    -1,       /* size of per-interpreter state of the module,\n"
    code += "                 or -1 if the module keeps state in global variables. */\n"
    code += "    " + mod_name + "_methods\n"
    code += "};\n\n"

    code += "PyMODINIT_FUNC PyInit_" + mod_name + "(void)\n"
    code += "{\n"
    code += "    PyObject *m;\n\n"
    code += "    m = PyModule_Create(&" + mod_name + "_module);\n"
    code += "    if (m == NULL)\n"
    code += "        return NULL;\n\n"
    code += "    return m;\n"
    code += "}"

    return code

def create_c_setup(mod_name, dependencies, compiler, flags):
    code  = "from setuptools import Extension, setup\n\n"

    flags = flags.replace('"','\\"')
    deps  = ", ".join("r\"{0}.c\"".format(d) for d in dependencies)
    code += "extension_mod = Extension(\"{0}\", [{1}], extra_compile_args = {2})\n\n".format(mod_name, deps, flags.strip().split())
    code += "setup(name = \"" + mod_name + "\", ext_modules=[extension_mod])"
    return code

