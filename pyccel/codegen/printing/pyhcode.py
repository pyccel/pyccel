# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=missing-function-docstring
"""
Handles printing the pyh header file
"""
from .codeprinter import CodePrinter
from pyccel.ast.core       import FunctionAddress
from pyccel.ast.datatypes  import NativeBool, NativeInteger
from pyccel.ast.datatypes  import NativeReal, NativeComplex
from pyccel.ast.datatypes  import CustomDataType

numpy_dtype_precision = {
        NativeBool() : {4 : 'bool'},
        NativeInteger() : {
            1 : 'int8',
            2 : 'int16',
            4 : 'int32',
            8 : 'int64'},
        NativeReal() : {
            4 : 'float32',
            8 : 'float64'},
        NativeComplex() : {
            4 : 'complex64',
            8 : 'complex128'}
        }

class PyhCodePrinter(CodePrinter):
    """A printer to convert pyccel expressions to strings of Python header code"""
    printmethod = "_pycode"
    language = "python"

    _default_settings = {
        'tabwidth': 4,
    }

    def __init__(self, output_folder, parser=None):
        self._parser = parser
        self._output_folder = output_folder
        super().__init__()

    def _indent_codestring(self, lines):
        tab = " "*self._default_settings['tabwidth']
        if lines == '':
            return lines
        else:
            # lines ends with \n
            return tab+lines.strip('\n').replace('\n','\n'+tab)+'\n'

    @classmethod
    def _var_annotation(self, var):
        dtype = var.dtype
        if isinstance(dtype, CustomDataType):
            name   = dtype.__class__.__name__
            prefix = dtype.prefix
            alias  = dtype.alias

            if alias is None:
                dtype_code = name.replace(prefix, '')
            else:
                dtype_code = alias
        else:
            dtype_code = numpy_dtype_precision[var.dtype][var.precision]

        if var.rank>0:
            dtype_code += '[{}]'.format(','.join([':']*var.rank))
        if var.rank>1:
            dtype_code += '(order={})'.format(var.order)
        return dtype_code

    def _function_address_annotation(self, expr):
        arg_types = ', '.join(self._get_annotation(a.var) for a in expr.arguments)
        res_types = ', '.join(self._get_annotation(r) for r in expr.results)
        return '({})({})'.format(res_types, arg_types)

    def _get_annotation(self, expr):
        if isinstance(expr, FunctionAddress):
            return self._function_address_annotation(expr)
        else:
            return self._var_annotation(expr)

    def _format_code(self, lines):
        return lines

    # .....................................................
    #               Print Functions
    # .....................................................

    def _print_Module(self, expr):
        output = "#$ header metavar compile_folder='{}'\n".format(self._output_folder)
        name = "#$ header metavar module_name='{}'\n".format(
                expr.name)
        # Collect functions which are not in an interface
        funcs = [f for f in expr.funcs if not (f is expr.init_func or f is expr.free_func)]
        func_headers = ''.join(self._print(func) for func in funcs)
        classes = ''.join(self._print(c) for c in expr.classes)

        return ''.join((output, name, func_headers, classes))

    def _print_FunctionDef(self, expr):
        type_name = 'method' if expr.cls_name is not None else 'function'

        name = expr.name
        arg_types = ', '.join(self._get_annotation(a.var) for a in expr.arguments)
        res_types = ', '.join(self._get_annotation(r) for r in expr.results)
        return '#$ header {type_name} {name}({arg_types}) results({result_types})\n'.format(
                type_name=type_name,
                name=name,
                arg_types=arg_types,
                result_types=res_types)

    def _print_ClassDef(self, expr):
        head = '#$ header class {}(public)\n'.format(expr.name)
        methods = ''.join(self._print(m) for m in expr.methods)
        return head+methods

    def _print_Variable(self, expr):
        return '#$ header variable {name} {dtype}\n'.format(
                name  = expr.name,
                dtype = self._get_annotation(expr))


#==============================================================================
def pyhcode(expr, **settings):
    """ Converts an expr to a string of Python header code

    Parameters
    ==========
    expr : Expr
        A Pyccel expression.
    """
    return PyhCodePrinter(settings).doprint(expr)
