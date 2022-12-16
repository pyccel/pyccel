# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=missing-function-docstring
import functools
from itertools import chain
import re

from pyccel.ast.basic     import ScopedNode

from pyccel.ast.builtins  import PythonRange, PythonComplex
from pyccel.ast.builtins  import PythonPrint, PythonType
from pyccel.ast.builtins  import PythonList, PythonTuple
from pyccel.ast.class_defs import ListClass

from pyccel.ast.core      import Declare, DottedFunctionCall, For, CodeBlock
from pyccel.ast.core      import FuncAddressDeclare, FunctionCall, FunctionCallArgument, FunctionDef
from pyccel.ast.core      import Deallocate
from pyccel.ast.core      import FunctionAddress, FunctionDefArgument
from pyccel.ast.core      import Assign, Import, AugAssign, AliasAssign
from pyccel.ast.core      import SeparatorComment
from pyccel.ast.core      import Module, AsName

from pyccel.ast.operators import PyccelAdd, PyccelMul, PyccelMinus, PyccelLt, PyccelGt
from pyccel.ast.operators import PyccelAssociativeParenthesis, PyccelMod
from pyccel.ast.operators import PyccelUnarySub, IfTernaryOperator

from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeComplex
from pyccel.ast.datatypes import NativeFloat, NativeTuple, datatype, default_precision

from pyccel.ast.internals import Slice, PrecomputedCode, get_final_precision

from pyccel.ast.literals  import LiteralTrue, LiteralFalse, LiteralImaginaryUnit, LiteralFloat
from pyccel.ast.literals  import LiteralString, LiteralInteger, Literal
from pyccel.ast.literals  import Nil

from pyccel.ast.mathext  import math_constants

from pyccel.ast.numpyext import NumpyFull, NumpyArray, NumpyArange
from pyccel.ast.numpyext import NumpyReal, NumpyImag, NumpyFloat

from pyccel.ast.utilities import expand_to_loops

from pyccel.ast.variable import IndexedElement
from pyccel.ast.variable import PyccelArraySize, Variable
from pyccel.ast.variable import DottedName
from pyccel.ast.variable import InhomogeneousTupleVariable, HomogeneousTupleVariable

from pyccel.ast.c_concepts import ObjectAddress, CMacro, CStringExpression

from pyccel.codegen.printing.codeprinter import CodePrinter

from pyccel.errors.errors   import Errors
from pyccel.errors.messages import (PYCCEL_RESTRICTION_TODO, INCOMPATIBLE_TYPEVAR_TO_FUNC,
                                    PYCCEL_RESTRICTION_IS_ISNOT, UNSUPPORTED_ARRAY_RANK)


errors = Errors()

# TODO: add examples

__all__ = ["CCodePrinter", "ccode"]

# dictionary mapping numpy function to (argument_conditions, C_function).
# Used in CCodePrinter._print_NumpyUfuncBase(self, expr)
numpy_ufunc_to_c_float = {
    'NumpyAbs'  : 'fabs',
    'NumpyFabs' : 'fabs',
    'NumpyMin'  : 'minval',
    'NumpyMax'  : 'maxval',
    'NumpyFloor': 'floor',  # TODO: might require special treatment with casting
    # ---
    'NumpyExp' : 'exp',
    'NumpyLog' : 'log',
    'NumpySqrt': 'sqrt',
    # ---
    'NumpySin'    : 'sin',
    'NumpyCos'    : 'cos',
    'NumpyTan'    : 'tan',
    'NumpyArcsin' : 'asin',
    'NumpyArccos' : 'acos',
    'NumpyArctan' : 'atan',
    'NumpyArctan2': 'atan2',
    'NumpySinh'   : 'sinh',
    'NumpyCosh'   : 'cosh',
    'NumpyTanh'   : 'tanh',
    'NumpyArcsinh': 'asinh',
    'NumpyArccosh': 'acosh',
    'NumpyArctanh': 'atanh',
}

numpy_ufunc_to_c_complex = {
    'NumpyAbs'  : 'cabs',
    'NumpyMin'  : 'minval',
    'NumpyMax'  : 'maxval',
    # ---
    'NumpyExp' : 'cexp',
    'NumpyLog' : 'clog',
    'NumpySqrt': 'csqrt',
    # ---
    'NumpySin'    : 'csin',
    'NumpyCos'    : 'ccos',
    'NumpyTan'    : 'ctan',
    'NumpyArcsin' : 'casin',
    'NumpyArccos' : 'cacos',
    'NumpyArctan' : 'catan',
    'NumpySinh'   : 'csinh',
    'NumpyCosh'   : 'ccosh',
    'NumpyTanh'   : 'ctanh',
    'NumpyArcsinh': 'casinh',
    'NumpyArccosh': 'cacosh',
    'NumpyArctanh': 'catanh',
}

# dictionary mapping Math function to (argument_conditions, C_function).
# Used in CCodePrinter._print_MathFunctionBase(self, expr)
# Math function ref https://docs.python.org/3/library/math.html
math_function_to_c = {
    # ---------- Number-theoretic and representation functions ------------
    'MathCeil'     : 'ceil',
    # 'MathComb'   : 'com' # TODO
    'MathCopysign': 'copysign',
    'MathFabs'   : 'fabs',
    'MathFloor'    : 'floor',
    # 'MathFmod'   : '???',  # TODO
    # 'MathRexp'   : '???'   TODO requires two output
    # 'MathFsum'   : '???',  # TODO
    # 'MathIsclose' : '???',  # TODO
    'MathIsfinite': 'isfinite', # int isfinite(real-floating x);
    'MathIsinf'   : 'isinf', # int isinf(real-floating x);
    'MathIsnan'   : 'isnan', # int isnan(real-floating x);
    # 'MathIsqrt'  : '???' TODO
    'MathLdexp'  : 'ldexp',
    # 'MathModf'  : '???' TODO return two value
    # 'MathPerm'  : '???' TODO
    # 'MathProd'  : '???' TODO
    'MathRemainder'  : 'remainder',
    'MathTrunc'  : 'trunc',

    # ----------------- Power and logarithmic functions -----------------------

    'MathExp'    : 'exp',
    'MathExpm1'  : 'expm1',
    'MathLog'    : 'log',      # take also an option arg [base]
    'MathLog1p'  : 'log1p',
    'MathLog2'  : 'log2',
    'MathLog10'  : 'log10',
    'MathPow'    : 'pow',
    'MathSqrt'   : 'sqrt',

    # --------------------- Trigonometric functions ---------------------------

    'MathAcos'   : 'acos',
    'MathAsin'   : 'asin',
    'MathAtan'   : 'atan',
    'MathAtan2'  : 'atan2',
    'MathCos'    : 'cos',
    # 'MathDist'  : '???', TODO
    'MathHypot'  : 'hypot',
    'MathSin'    : 'sin',
    'MathTan'    : 'tan',


    # -------------------------- Hyperbolic functions -------------------------

    'MathAcosh'  : 'acosh',
    'MathAsinh'  : 'asinh',
    'MathAtanh'  : 'atanh',
    'MathCosh'   : 'cosh',
    'MathSinh'   : 'sinh',
    'MathTanh'   : 'tanh',

    # --------------------------- Special functions ---------------------------

    'MathErf'    : 'erf',
    'MathErfc'   : 'erfc',
    'MathGamma'  : 'tgamma',
    'MathLgamma' : 'lgamma',

    # --------------------------- internal functions --------------------------
    'MathFactorial' : 'pyc_factorial',
    'MathGcd'       : 'pyc_gcd',
    'MathDegrees'   : 'pyc_degrees',
    'MathRadians'   : 'pyc_radians',
    'MathLcm'       : 'pyc_lcm',
}

c_library_headers = (
    "complex",
    "ctype",
    "float",
    "math",
    "stdarg",
    "stdbool",
    "stddef",
    "stdint",
    "stdio",
    "stdlib",
    "string",
    "tgmath",
    "inttypes",
)

dtype_registry = {('float',8)   : 'double',
                  ('float',4)   : 'float',
                  ('complex',8) : 'double complex',
                  ('complex',4) : 'float complex',
                  ('int',4)     : 'int32_t',
                  ('int',8)     : 'int64_t',
                  ('int',2)     : 'int16_t',
                  ('int',1)     : 'int8_t',
                  ('bool',4)    : 'bool'}

ndarray_type_registry = {
                  ('float',8)   : 'nd_double',
                  ('float',4)   : 'nd_float',
                  ('complex',8) : 'nd_cdouble',
                  ('complex',4) : 'nd_cfloat',
                  ('int',8)     : 'nd_int64',
                  ('int',4)     : 'nd_int32',
                  ('int',2)     : 'nd_int16',
                  ('int',1)     : 'nd_int8',
                  ('bool',4)    : 'nd_bool'}

list_type_registry = {
                  ('float',8)   : 'lst_double',
                  ('float',4)   : 'lst_float',
                  ('complex',8) : 'lst_complex',
                  ('int',8)     : 'lst_int64',
                  ('int',4)     : 'lst_int32',
                  ('int',2)     : 'lst_int16',
                  ('int',1)     : 'lst_int8',
                  ('bool',4)    : 'lst_bool'}

type_to_format = {('float',8)   : '%.12lf',
                  ('float',4)   : '%.12f',
                  ('complex',8) : '(%.12lf + %.12lfj)',
                  ('complex',4) : '(%.12f + %.12fj)',
                  ('int',4)     : '%d',
                  ('int',8)     : LiteralString("%") + CMacro('PRId64'),
                  ('int',2)     : LiteralString("%") + CMacro('PRId16'),
                  ('int',1)     : LiteralString("%") + CMacro('PRId8'),
                  ('bool',4)    : '%s',
                  ('string', 0) : '%s'}

import_dict = {'omp_lib' : 'omp' }

c_imports = {n : Import(n, Module(n, (), ())) for n in
                ['stdlib',
                 'math',
                 'string',
                 'ndarrays',
                 'math',
                 'complex',
                 'stdint',
                 'pyc_math_c',
                 'stdio',
                 "inttypes",
                 'stdbool',
                 'assert',
                 'numpy_c',
                 'lists']}

class CCodePrinter(CodePrinter):
    """A printer to convert python expressions to strings of c code"""
    printmethod = "_ccode"
    language = "C"

    _default_settings = {
        'tabwidth': 4,
    }

    def __init__(self, filename, prefix_module = None):

        errors.set_target(filename, 'file')

        super().__init__()
        self.prefix_module = prefix_module
        self._additional_imports = {'stdlib':c_imports['stdlib']}
        self._additional_code = ''
        self._additional_args = []
        self._temporary_args = []
        self._current_module = None
        self._in_header = False
        # Dictionary linking optional variables to their
        # temporary counterparts which provide allocated
        # memory
        # Key is optional variable
        self._optional_partners = {}

    def get_additional_imports(self):
        """return the additional imports collected in printing stage"""
        return self._additional_imports.keys()

    def add_import(self, import_obj):
        if import_obj.source not in self._additional_imports:
            self._additional_imports[import_obj.source] = import_obj

    def _get_statement(self, codestring):
        return "%s;\n" % codestring

    def _get_comment(self, text):
        return "// {0}\n".format(text)

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _flatten_list(self, irregular_list):
        if isinstance(irregular_list, (PythonList, PythonTuple)):
            f_list = [element for item in irregular_list for element in self._flatten_list(item)]
            return f_list
        else:
            return [irregular_list]

    #========================== Numpy Elements ===============================#
    def copy_NumpyArray_Data(self, expr):
        """ print the assignment of a NdArray or a homogeneous tuple

        parameters
        ----------
            expr : PyccelAstNode
                The Assign Node used to get the lhs and rhs
        Return
        ------
            String
                Return a str that contains the declaration of a dummy data_buffer
                       and a call to an operator which copies it to an NdArray struct
        """
        rhs = expr.rhs
        lhs = expr.lhs
        if rhs.rank == 0:
            raise NotImplementedError(str(expr))
        dummy_array_name = self.scope.get_new_name('array_dummy')
        declare_dtype = self.find_in_dtype_registry(self._print(rhs.dtype), rhs.precision)
        dtype = self.find_in_ndarray_type_registry(self._print(rhs.dtype), rhs.precision)
        arg = rhs.arg if isinstance(rhs, NumpyArray) else rhs

        self.add_import(c_imports['string'])
        if isinstance(arg, Variable):
            arg = self._print(arg)
            cpy_data = "memcpy({0}.{2}, {1}.{2}, {0}.buffer_size);\n".format(lhs, arg, dtype)
            return '%s' % (cpy_data)
        else :
            if arg.rank > 1:
                # flattening the args to use them in C initialization.
                arg = self._flatten_list(arg)
            arg = ', '.join(self._print(i) for i in arg)
            dummy_array = "%s %s[] = {%s};\n" % (declare_dtype, dummy_array_name, arg)
            cpy_data = "memcpy({0}.{2}, {1}, {0}.buffer_size);\n".format(self._print(lhs), dummy_array_name, dtype)
            return  '%s%s' % (dummy_array, cpy_data)

    def arrayFill(self, expr):
        """ print the assignment of a NdArray

        parameters
        ----------
            expr : PyccelAstNode
                The Assign Node used to get the lhs and rhs
        Return
        ------
            String
                Return a str that contains a call to the C function array_fill,
        """
        rhs = expr.rhs
        lhs = expr.lhs
        code_init = ''
        declare_dtype = self.find_in_dtype_registry(self._print(rhs.dtype), rhs.precision)

        if rhs.fill_value is not None:
            if isinstance(rhs.fill_value, Literal):
                code_init += 'array_fill(({0}){1}, {2});\n'.format(declare_dtype, self._print(rhs.fill_value), self._print(lhs))
            else:
                code_init += 'array_fill({0}, {1});\n'.format(self._print(rhs.fill_value), self._print(lhs))
        return code_init

    def _init_stack_array(self, expr):
        """ return a string which handles the assignment of a stack ndarray

        Parameters
        ----------
            expr : PyccelAstNode
                The Assign Node used to get the lhs and rhs
        Returns
        -------
            buffer_array : str
                String initialising the stack (C) array which stores the data
            array_init   : str
                String containing the rhs of the initialization of a stack array
        """
        var = expr
        dtype_str = self._print(var.dtype)
        dtype = self.find_in_dtype_registry(dtype_str, var.precision)
        np_dtype = self.find_in_ndarray_type_registry(dtype_str, var.precision)
        shape = ", ".join(self._print(i) for i in var.alloc_shape)
        tot_shape = self._print(functools.reduce(
            lambda x,y: PyccelMul(x,y,simplify=True), var.alloc_shape))
        declare_dtype = self.find_in_dtype_registry('int', 8)

        dummy_array_name = self.scope.get_new_name('array_dummy')
        buffer_array = "{dtype} {name}[{size}];\n".format(
                dtype = dtype,
                name  = dummy_array_name,
                size  = tot_shape)
        shape_init = "({declare_dtype}[]){{{shape}}}".format(declare_dtype=declare_dtype, shape=shape)
        strides_init = "({declare_dtype}[{length}]){{0}}".format(declare_dtype=declare_dtype, length=len(var.shape))
        array_init = ' = (t_ndarray){{\n.{0}={1},\n .shape={2},\n .strides={3},\n '
        array_init += '.nd={4},\n .type={0},\n .is_view={5}\n}};\n'
        array_init = array_init.format(np_dtype, dummy_array_name,
                    shape_init, strides_init, len(var.shape), 'false')
        array_init += 'stack_array_init(&{})'.format(self._print(var))
        self.add_import(c_imports['ndarrays'])
        return buffer_array, array_init

    def _handle_inline_func_call(self, expr):
        """ Print a function call to an inline function
        """
        func = expr.funcdef
        body = func.body

        for b in body.body:
            if isinstance(b, ScopedNode):
                b.scope.update_parent_scope(self.scope, is_loop=True)

        # Print any arguments using the same inline function
        # As the function definition is modified directly this function
        # cannot be called recursively with the same FunctionDef
        args = []
        for a in expr.args:
            if a.is_user_of(func):
                code = PrecomputedCode(self._print(a))
                args.append(code)
            else:
                args.append(a.value)

        new_local_vars = [self.scope.get_temporary_variable(v) \
                            for v in func.local_vars]

        parent_assign = expr.get_direct_user_nodes(lambda x: isinstance(x, Assign))
        if parent_assign:
            results = dict(zip(func.results, parent_assign[0].lhs))
            orig_res_vars = list(results.keys())
            new_res_vars  = self._temporary_args
            new_res_vars = [a.obj if isinstance(a, ObjectAddress) else a for a in new_res_vars]
            self._temporary_args = []
            body.substitute(orig_res_vars, new_res_vars)

        # Replace the arguments in the code
        func.swap_in_args(args, new_local_vars)

        func.remove_presence_checks()

        # Collect code but strip empty end
        body_code = self._print(body)
        code_lines = body_code.split('\n')[:-1]
        return_regex = re.compile(r'\breturn\b')
        has_results = [return_regex.search(l) is not None for l in code_lines]

        if len(func.results) == 0 and not any(has_results):
            code = body_code
        else:
            result_idx = has_results.index(True)
            result_line = code_lines[result_idx]

            body_code = '\n'.join(code_lines[:result_idx])+'\n'

            if len(func.results) != 1:
                code = body_code
            else:
                self._additional_code += body_code
                # Strip return and ; from return statement
                code = result_line[7:-1]

        # Put back original arguments
        func.reinstate_presence_checks()
        func.swap_out_args()
        if parent_assign:
            body.substitute(new_res_vars, orig_res_vars)

        if func.global_vars or func.global_funcs:
            mod = func.get_direct_user_nodes(lambda x: isinstance(x, Module))[0]
            self.add_import(Import(mod.name, [AsName(v, v.name) \
                for v in (*func.global_vars, *func.global_funcs)]))
            for v in (*func.global_vars, *func.global_funcs):
                self.scope.insert_symbol(v.name)

        for b in body.body:
            if isinstance(b, ScopedNode):
                b.scope.update_parent_scope(func.scope, is_loop=True)

        return code

    # ============ Elements ============ #
    
    def compound_stmt(self, expr, compound_type=None):
        if compound_type == None:
            compound_type = self.find_in_dtype_registry(self._print(expr.dtype), expr.precision)
        args = ', \n'.join(self._print(arg) for arg in expr.args)
        stmt = f'({compound_type}[]){{\n{args}\n}}'
        return stmt

    def fill_list(self, expr):
        lhs = expr.lhs
        rhs = expr.rhs

        lhs  = self._print(lhs)
        stmt = self.compound_stmt(rhs, None if rhs.rank == 1 else 't_list *')
        return "memcpy({}->elements, {}, {}->size * tSizes[{}->type]);\n".format(lhs, stmt, lhs, lhs)

    def _print_PythonList(self, expr):
        if expr.rank == 1:
            lst_type = self.find_in_list_type_registry(self._print(expr.dtype), expr.precision)
            elements = self.compound_stmt(expr)
        else:
            lst_type = "lst_list"
            elements = self.compound_stmt(expr, 't_list*')
        return "allocate_list({}, {}, {})".format(expr.shape[0], lst_type, elements)

    def _print_PythonListAppend(self, expr):
        lst = expr.list
        arg = expr.args[0].value

        if arg.rank == 0 and lst.rank == 1 and not isinstance(arg, Variable):
            compound_type = self.find_in_dtype_registry(self._print(arg.dtype), arg.precision)
            arg = f"({compound_type}[]){{{self._print(arg)}}}"
        elif isinstance(arg, Variable):
            arg = ObjectAddress(arg)

        return "append({}, {});\n".format(self._print(lst), self._print(arg))

    def _print_PythonListSort(self, expr):
        lst = expr.list
        rev = expr.reverse
        return "lst_sort({}, {});\n".format(
                    self._print(lst),
                    self._print(rev))

    def _print_PythonListClear(self, expr):
        return "clear({});\n".format(self._print(expr.list))

    def _print_PythonListExtend(self, expr):
        lst = expr.list
        arg = expr.args[0].value
        return "extend({}, {});\n".format(self._print(lst), self._print(arg))

    def _print_PythonListCount(self, expr):
        lst = expr.list
        arg = expr.args[0].value

        if arg.rank == 0 and not isinstance(arg, Variable):
            compound_type = self.find_in_dtype_registry(self._print(arg.dtype), arg.precision)
            arg = f"({compound_type}[]){{{self._print(arg)}}}"
        elif isinstance(arg, Variable):
            arg = ObjectAddress(arg)

        return "count({}, {})".format(self._print(lst), self._print(arg))

    def _print_PythonListInsert(self, expr):
        lst = expr.list
        idx = expr.args[0].value
        obj = expr.args[1].value

        if obj.rank == 0 and not isinstance(obj, Variable):
            compound_type = self.find_in_dtype_registry(self._print(obj.dtype), obj.precision)
            obj = f"({compound_type}[]){{{self._print(obj)}}}"
        elif isinstance(obj, Variable):
            obj = ObjectAddress(obj)

        return "insert({}, {}, {});\n".format(
            self._print(lst), 
            self._print(idx), 
            self._print(obj))

    def _print_PythonListPop(self, expr):
        lst = expr.list
        arg = expr.args[0].value

        if lst.rank == 1:
            dtype = self.find_in_dtype_registry(self._print(lst.dtype), lst.precision)
            prefix = "*({} *)".format(dtype)
        else:
            prefix = "(t_list *)".format()
        return "{}pop({}, {})".format(prefix, self._print(lst), self._print(arg))

    def _print_PythonListReverse(self, expr):
        return "reverse({});\n".format(self._print(expr.list))

    def _print_PythonListRemove(self, expr):
        lst = expr.list
        arg = expr.args[0].value

        if arg.rank == 0 and not isinstance(arg, Variable):
            compound_type = self.find_in_dtype_registry(self._print(arg.dtype), arg.precision)
            arg = f"({compound_type}[]){{{self._print(arg)}}}"
        elif isinstance(arg, Variable):
            arg = ObjectAddress(arg)

        return "lst_remove({}, {});\n".format(self._print(lst), self._print(arg))

    def _print_PythonAbs(self, expr):
        if expr.arg.dtype is NativeFloat():
            self.add_import(c_imports['math'])
            func = "fabs"
        elif expr.arg.dtype is NativeComplex():
            self.add_import(c_imports['complex'])
            func = "cabs"
        else:
            func = "labs"
        return "{}({})".format(func, self._print(expr.arg))

    def _print_PythonMin(self, expr):
        arg = expr.args[0]
        if arg.dtype is NativeFloat() and len(arg) == 2:
            self.add_import(c_imports['math'])
            return "fmin({}, {})".format(self._print(arg[0]),
                                         self._print(arg[1]))
        elif arg.dtype is NativeInteger() and len(arg) == 2:
            arg1 = self.scope.get_temporary_variable(NativeInteger())
            arg2 = self.scope.get_temporary_variable(NativeInteger())
            assign1 = Assign(arg1, arg[0])
            assign2 = Assign(arg2, arg[1])
            self._additional_code += self._print(assign1)
            self._additional_code += self._print(assign2)
            return f"({arg1} < {arg2} ? {arg1} : {arg2})"
        else:
            return errors.report("min in C is only supported for 2 scalar arguments", symbol=expr,
                    severity='fatal')

    def _print_PythonMax(self, expr):
        arg = expr.args[0]
        if arg.dtype is NativeFloat() and len(arg) == 2:
            self.add_import(c_imports['math'])
            return "fmax({}, {})".format(self._print(arg[0]),
                                         self._print(arg[1]))
        elif arg.dtype is NativeInteger() and len(arg) == 2:
            arg1 = self.scope.get_temporary_variable(NativeInteger())
            arg2 = self.scope.get_temporary_variable(NativeInteger())
            assign1 = Assign(arg1, arg[0])
            assign2 = Assign(arg2, arg[1])
            self._additional_code += self._print(assign1)
            self._additional_code += self._print(assign2)
            return f"({arg1} > {arg2} ? {arg1} : {arg2})"
        else:
            return errors.report("max in C is only supported for 2 scalar arguments", symbol=expr,
                    severity='fatal')

    def _print_SysExit(self, expr):
        code = ""
        if expr.status.dtype is not NativeInteger() or expr.status.rank > 0:
            print_arg = FunctionCallArgument(expr.status)
            code = self._print(PythonPrint((print_arg, ), file="stderr"))
            arg = "1"
        else:
            arg = self._print(expr.status)
        return f"{code}exit({arg});\n"

    def _print_PythonFloat(self, expr):
        value = self._print(expr.arg)
        type_name = self.find_in_dtype_registry('float', expr.precision)
        return '({0})({1})'.format(type_name, value)

    def _print_PythonInt(self, expr):
        self.add_import(c_imports['stdint'])
        value = self._print(expr.arg)
        type_name = self.find_in_dtype_registry('int', expr.precision)
        return '({0})({1})'.format(type_name, value)

    def _print_PythonBool(self, expr):
        value = self._print(expr.arg)
        return '({} != 0)'.format(value)

    def _print_Literal(self, expr):
        return repr(expr.python_value)

    def _print_LiteralInteger(self, expr):
        if isinstance(expr, LiteralInteger) and get_final_precision(expr) == 8:
            self.add_import(c_imports['stdint'])
            return f"INT64_C({repr(expr.python_value)})"
        return repr(expr.python_value)

    def _print_LiteralFloat(self, expr):
        if isinstance(expr, LiteralFloat) and get_final_precision(expr) == 4:
            return f"{repr(expr.python_value)}f"
        return repr(expr.python_value)

    def _print_LiteralComplex(self, expr):
        if expr.real == LiteralFloat(0):
            return self._print(PyccelAssociativeParenthesis(PyccelMul(expr.imag, LiteralImaginaryUnit())))
        else:
            return self._print(PyccelAssociativeParenthesis(PyccelAdd(expr.real,
                            PyccelMul(expr.imag, LiteralImaginaryUnit()))))

    def _print_PythonComplex(self, expr):
        if expr.is_cast:
            value = self._print(expr.internal_var)
        else:
            value = self._print(PyccelAssociativeParenthesis(PyccelAdd(expr.real,
                            PyccelMul(expr.imag, LiteralImaginaryUnit()))))
        type_name = self.find_in_dtype_registry('complex', expr.precision)
        return '({0})({1})'.format(type_name, value)

    def _print_LiteralImaginaryUnit(self, expr):
        self.add_import(c_imports['complex'])
        return '_Complex_I'

    def _print_PythonLen(self, expr):
        var = expr.arg
        if isinstance(var, PythonList) or (isinstance(var, Variable) and var.cls_base == ListClass):
            return f"{var}->size"
        if var.rank > 0:
            return self._print(var.shape[0])
        else:
            return errors.report("PythonLen not implemented for type {}\n".format(type(expr.arg)) +
                    PYCCEL_RESTRICTION_TODO,
                    symbol = expr, severity='fatal')

    def _print_Header(self, expr):
        return ''

    def _print_ModuleHeader(self, expr):
        self.set_scope(expr.module.scope)
        self._in_header = True
        name = expr.module.name
        if isinstance(name, AsName):
            name = name.name
        # TODO: Add classes and interfaces
        funcs = '\n'.join('{};'.format(self.function_signature(f)) for f in expr.module.funcs)

        global_variables = ''.join(['extern '+self._print(d) for d in expr.module.declarations if not d.variable.is_private])

        # Print imports last to be sure that all additional_imports have been collected
        imports = [*expr.module.imports, *self._additional_imports.values()]
        imports = ''.join(self._print(i) for i in imports)

        self._in_header = False
        self.exit_scope()
        return ('#ifndef {name}_H\n'
                '#define {name}_H\n\n'
                '{imports}\n'
                '{variables}\n'
                #'{classes}\n'
                '{funcs}\n'
                #'{interfaces}\n'
                '#endif // {name}_H\n').format(
                        name    = name.upper(),
                        imports = imports,
                        variables = global_variables,
                        funcs   = funcs)

    def _print_Module(self, expr):
        self.set_scope(expr.scope)
        self._current_module = expr.name
        body    = ''.join(self._print(i) for i in expr.body)

        global_variables = ''.join([self._print(d) for d in expr.declarations])

        # Print imports last to be sure that all additional_imports have been collected
        imports = [Import(expr.name, Module(expr.name,(),())), *self._additional_imports.values()]
        imports = ''.join(self._print(i) for i in imports)

        code = ('{imports}\n'
                '{variables}\n'
                '{body}\n').format(
                        imports   = imports,
                        variables = global_variables,
                        body      = body)

        self.exit_scope()
        return code

    def _print_Break(self, expr):
        return 'break;\n'

    def _print_Continue(self, expr):
        return 'continue;\n'

    def _print_While(self, expr):
        self.set_scope(expr.scope)
        body = self._print(expr.body)
        self.exit_scope()
        cond = self._print(expr.test)
        return 'while({condi})\n{{\n{body}}}\n'.format(condi = cond, body = body)

    def _print_If(self, expr):
        lines = []
        for i, (c, e) in enumerate(expr.blocks):
            var = self._print(e)
            if i == 0:
                lines.append("if (%s)\n{\n" % self._print(c))
            elif i == len(expr.blocks) - 1 and isinstance(c, LiteralTrue):
                lines.append("else\n{\n")
            else:
                lines.append("else if (%s)\n{\n" % self._print(c))
            lines.append("%s}\n" % var)
        return "".join(lines)

    def _print_IfTernaryOperator(self, expr):
        cond = self._print(expr.cond)
        value_true = self._print(expr.value_true)
        value_false = self._print(expr.value_false)
        return '{cond} ? {true} : {false}'.format(cond = cond, true =value_true, false = value_false)

    def _print_LiteralTrue(self, expr):
        return '1'

    def _print_LiteralFalse(self, expr):
        return '0'

    def _print_PyccelAnd(self, expr):
        args = [self._print(a) for a in expr.args]
        return ' && '.join(a for a in args)

    def _print_PyccelOr(self, expr):
        args = [self._print(a) for a in expr.args]
        return ' || '.join(a for a in args)

    def _print_PyccelEq(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} == {1}'.format(lhs, rhs)

    def _print_PyccelNe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} != {1}'.format(lhs, rhs)

    def _print_PyccelLt(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} < {1}'.format(lhs, rhs)

    def _print_PyccelLe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} <= {1}'.format(lhs, rhs)

    def _print_PyccelGt(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} > {1}'.format(lhs, rhs)

    def _print_PyccelGe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} >= {1}'.format(lhs, rhs)

    def _print_PyccelNot(self, expr):
        a = self._print(expr.args[0])
        return '!{}'.format(a)

    def _print_PyccelMod(self, expr):
        self.add_import(c_imports['math'])
        self.add_import(c_imports['pyc_math_c'])

        first = self._print(expr.args[0])
        second = self._print(expr.args[1])

        if expr.dtype is NativeInteger():
            return "pyc_modulo({n}, {base})".format(n=first, base=second)

        if expr.args[0].dtype is NativeInteger():
            first = self._print(NumpyFloat(expr.args[0]))
        if expr.args[1].dtype is NativeInteger():
            second = self._print(NumpyFloat(expr.args[1]))
        return "pyc_fmodulo({n}, {base})".format(n=first, base=second)

    def _print_PyccelPow(self, expr):
        b = expr.args[0]
        e = expr.args[1]

        if expr.dtype is NativeComplex():
            b = self._print(b if b.dtype is NativeComplex() else PythonComplex(b))
            e = self._print(e if e.dtype is NativeComplex() else PythonComplex(e))
            self.add_import(c_imports['complex'])
            return 'cpow({}, {})'.format(b, e)

        self.add_import(c_imports['math'])
        b = self._print(b if b.dtype is NativeFloat() else NumpyFloat(b))
        e = self._print(e if e.dtype is NativeFloat() else NumpyFloat(e))
        code = 'pow({}, {})'.format(b, e)
        if expr.dtype is NativeInteger():
            dtype = self._print(expr.dtype)
            prec  = expr.precision
            cast_type = self.find_in_dtype_registry(dtype, prec)
            return '({}){}'.format(cast_type, code)
        return code

    def _print_Import(self, expr):
        if expr.ignore:
            return ''
        if isinstance(expr.source, AsName):
            source = expr.source.name
        else:
            source = expr.source
        if isinstance(source, DottedName):
            source = source.name[-1]
        else:
            source = self._print(source)

        # Get with a default value is not used here as it is
        # slower and on most occasions the import will not be in the
        # dictionary
        if source in import_dict: # pylint: disable=consider-using-get
            source = import_dict[source]

        if source is None:
            return ''
        if expr.source in c_library_headers:
            return '#include <{0}.h>\n'.format(source)
        else:
            return '#include "{0}.h"\n'.format(source)

    def _print_LiteralString(self, expr):
        format_str = format(expr.arg)
        format_str = format_str.replace("\\", "\\\\")\
                               .replace('\a', '\\a')\
                               .replace('\b', '\\b')\
                               .replace('\f', '\\f')\
                               .replace("\n", "\\n")\
                               .replace('\r', '\\r')\
                               .replace('\t', '\\t')\
                               .replace('\v', '\\v')\
                               .replace('"', '\\"')\
                               .replace("'", "\\'")
        return '"{}"'.format(format_str)

    def get_print_format_and_arg(self, var):
        try:
            arg_format = type_to_format[(self._print(var.dtype), get_final_precision(var))]
        except KeyError:
            errors.report("{} type is not supported currently".format(var.dtype), severity='fatal')
        if var.dtype is NativeComplex():
            arg = '{}, {}'.format(self._print(NumpyReal(var)), self._print(NumpyImag(var)))
        elif var.dtype is NativeBool():
            arg = '{} ? "True" : "False"'.format(self._print(var))
        else:
            arg = self._print(var)
        return arg_format, arg

    def _print_CStringExpression(self, expr):
        return "".join(self._print(e) for e in expr.get_flat_expression_list())

    def _print_CMacro(self, expr):
        return str(expr.macro)

    def extract_function_call_results(self, expr):
        tmp_list = [self.scope.get_temporary_variable(a.dtype) for a in expr.funcdef.results]
        return tmp_list

    def _print_PythonPrint(self, expr):
        self.add_import(c_imports['stdio'])
        self.add_import(c_imports['inttypes'])

        end = '\n'
        sep = ' '
        code = ''
        empty_end = FunctionCallArgument(LiteralString(''), 'end')
        space_end = FunctionCallArgument(LiteralString(' '), 'end')
        empty_sep = FunctionCallArgument(LiteralString(''), 'sep')
        kwargs = [f for f in expr.expr if f.has_keyword]
        for f in kwargs:
            if f.keyword == 'sep'      :   sep = str(f.value)
            elif f.keyword == 'end'    :   end = str(f.value)
            else: errors.report("{} not implemented as a keyworded argument".format(f.keyword), severity='fatal')
        args_format = []
        args = []
        orig_args = [f for f in expr.expr if not f.has_keyword]

        def formatted_args_to_printf(args_format, args, end):
            args_format = CStringExpression(sep).join(args_format)
            args_format += end
            args_format = self._print(args_format)
            args_code = ', '.join([args_format, *args])
            if expr.file == 'stderr':
                return f"fprintf(stderr, {args_code});\n"
            return f"printf({args_code});\n"

        if len(orig_args) == 0:
            return formatted_args_to_printf(args_format, args, end)

        tuple_start = FunctionCallArgument(LiteralString('('))
        tuple_sep   = LiteralString(', ')
        tuple_end   = FunctionCallArgument(LiteralString(')'))

        for i, f in enumerate(orig_args):
            f = f.value
            if isinstance(f, (InhomogeneousTupleVariable, PythonTuple)):
                if args_format:
                    code += formatted_args_to_printf(args_format, args, sep)
                    args_format = []
                    args = []
                args = [FunctionCallArgument(print_arg) for tuple_elem in f for print_arg in (tuple_elem, tuple_sep)][:-1]
                if len(f) == 1:
                    args.append(FunctionCallArgument(LiteralString(',')))
                if i + 1 == len(orig_args):
                    end_of_tuple = FunctionCallArgument(LiteralString(end), 'end')
                else:
                    end_of_tuple = FunctionCallArgument(LiteralString(sep), 'end')
                code += self._print(PythonPrint([tuple_start, *args, tuple_end, empty_sep, end_of_tuple]))
                args = []
                continue
            if isinstance(f, PythonType):
                f = f.print_string
            elif isinstance(f, PythonList) or (isinstance(f, Variable) and f.is_list):
                if args_format:
                    code += formatted_args_to_printf(args_format, args, sep)
                    args_format = []
                    args = []
                new_line = '1' if i+1 == len(orig_args) else '0'
                code += f"print_list({self._print(f)}, {new_line});\n"
            elif isinstance(f, FunctionCall) and isinstance(f.dtype, NativeTuple):
                tmp_list = self.extract_function_call_results(f)
                tmp_arg_format_list = []
                for a in tmp_list:
                    arg_format, arg = self.get_print_format_and_arg(a)
                    tmp_arg_format_list.append(arg_format)
                    args.append(arg)
                tmp_arg_format_list = CStringExpression(', ').join(tmp_arg_format_list)
                args_format.append(CStringExpression('(', tmp_arg_format_list, ')'))
                assign = Assign(tmp_list, f)
                self._additional_code += self._print(assign)
            elif f.rank > 0:
                if args_format:
                    code += formatted_args_to_printf(args_format, args, sep)
                    args_format = []
                    args = []
                for_index = self.scope.get_temporary_variable(NativeInteger(), name = 'i')
                max_index = PyccelMinus(f.shape[0], LiteralInteger(1), simplify = True)
                for_range = PythonRange(max_index)
                print_body = [ FunctionCallArgument(f[for_index]) ]
                if f.rank == 1:
                    print_body.append(space_end)

                for_body  = [PythonPrint(print_body, file=expr.file)]
                for_scope = self.scope.create_new_loop_scope()
                for_loop  = For(for_index, for_range, for_body, scope=for_scope)
                for_end   = FunctionCallArgument(LiteralString(']'+end if i == len(orig_args)-1 else ']'), keyword='end')

                body = CodeBlock([PythonPrint([ FunctionCallArgument(LiteralString('[')), empty_end],
                                                file=expr.file),
                                  for_loop,
                                  PythonPrint([ FunctionCallArgument(f[max_index]), for_end],
                                                file=expr.file)],
                                 unravelled = True)
                code += self._print(body)
            else:
                arg_format, arg = self.get_print_format_and_arg(f)
                args_format.append(arg_format)
                args.append(arg)
        if args_format:
            code += formatted_args_to_printf(args_format, args, end)
        return code

    def find_in_dtype_registry(self, dtype, prec):
        if prec == -1:
            prec = default_precision[dtype]
        try :
            return dtype_registry[(dtype, prec)]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO,
                    symbol = "{}[kind = {}]".format(dtype, prec),
                    severity='fatal')

    def find_in_list_type_registry(self, dtype, prec):
        if prec == -1:
            prec = default_precision[dtype]
        try:
            return list_type_registry[(dtype, prec)]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO,
                    symbol = "{}[kind = {}]".format(dtype, prec),
                    severity = 'fatal')

    def find_in_ndarray_type_registry(self, dtype, prec):
        if prec == -1:
            prec = default_precision[dtype]
        try :
            return ndarray_type_registry[(dtype, prec)]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO,
                    symbol = "{}[kind = {}]".format(dtype, prec),
                    severity='fatal')

    def get_declare_type(self, expr):
        dtype = self._print(expr.dtype)
        prec  = expr.precision
        rank  = expr.rank
        if isinstance(expr.dtype, NativeInteger):
            self.add_import(c_imports['stdint'])
        dtype = self.find_in_dtype_registry(dtype, prec)
        if rank > 0:
            if isinstance(expr, PythonList) or (isinstance(expr, Variable) and expr.is_list):
                self.add_import(c_imports['lists'])
                dtype = 't_list'
            elif expr.is_ndarray or isinstance(expr, HomogeneousTupleVariable):
                if expr.rank > 15:
                    errors.report(UNSUPPORTED_ARRAY_RANK, symbol=expr, severity='fatal')
                self.add_import(c_imports['ndarrays'])
                dtype = 't_ndarray'
            else:
                errors.report(PYCCEL_RESTRICTION_TODO+' (rank>0)', symbol=expr, severity='fatal')

        if (self.stored_in_c_pointer(expr) or 
            isinstance(expr, PythonList) or (isinstance(expr, Variable) and expr.is_list)):
            return '{0} *'.format(dtype)
        else:
            return '{0} '.format(dtype)

    def _print_FuncAddressDeclare(self, expr):
        args = list(expr.arguments)
        if len(expr.results) == 1:
            ret_type = self.get_declare_type(expr.results[0])
        elif len(expr.results) > 1:
            ret_type = self._print(datatype('int')) + ' '
            args += [a.clone(name = a.name, memory_handling='alias') for a in expr.results]
        else:
            ret_type = self._print(datatype('void')) + ' '
        name = expr.name
        if not args:
            arg_code = 'void'
        else:
            # TODO: extract informations needed for printing in case of function argument which itself has a function argument
            arg_code = ', '.join('{}'.format(self._print_FuncAddressDeclare(i))
                        if isinstance(i, FunctionAddress) else '{0}{1}'.format(self.get_declare_type(i), i)
                        for i in args)
        return '{}(*{})({});\n'.format(ret_type, name, arg_code)

    def _print_Declare(self, expr):
        if isinstance(expr.variable, InhomogeneousTupleVariable):
            return ''.join(self._print_Declare(Declare(v.dtype,v,intent=expr.intent, static=expr.static)) for v in expr.variable)

        declaration_type = self.get_declare_type(expr.variable)
        variable = self._print(expr.variable.name)

        if expr.variable.is_stack_array:
            preface, init = self._init_stack_array(expr.variable,)
        elif declaration_type == 't_list *':
            preface = ''
            init    = ' = NULL'
        elif declaration_type == 't_ndarray ' and not self._in_header:
            preface = ''
            init    = ' = {.shape = NULL}'
        else:
            preface = ''
            init    = ''

        declaration = '{dtype}{var}{init};\n'.format(
                            dtype = declaration_type,
                            var   = variable,
                            init  = init)

        return preface + declaration

    def _print_NativeBool(self, expr):
        self.add_import(c_imports['stdbool'])
        return 'bool'

    def _print_NativeInteger(self, expr):
        return 'int'

    def _print_NativeFloat(self, expr):
        return 'float'

    def _print_NativeVoid(self, expr):
        return 'void'

    def _print_NativeComplex(self, expr):
        self.add_import(c_imports['complex'])
        return 'complex'
    def _print_NativeString(self, expr):
        return 'string'

    def function_signature(self, expr, print_arg_names = True):
        """Extract from function definition all the information
        (name, input, output) needed to create the signature

        Parameters
        ----------
        expr            : FunctionDef
            the function defintion

        print_arg_names : Bool
            default value True and False when we don't need to print
            arguments names

        Return
        ------
        String
            Signature of the function
        """
        if len(expr.results) > 1:
            self._additional_args.append(expr.results)
        args = list(expr.arguments)
        if len(expr.results) == 1:
            ret_type = self.get_declare_type(expr.results[0])
        elif len(expr.results) > 1:
            ret_type = self._print(datatype('int')) + ' '
            args += [FunctionDefArgument(a) for a in expr.results]
        else:
            ret_type = self._print(datatype('void')) + ' '
        name = expr.name
        if not args:
            arg_code = 'void'
        else:
            def get_var_arg(arg, var):
                code = "const " * var.is_const
                code += self.get_declare_type(var)
                code += arg.name * print_arg_names
                return code

            var_list = [a.var for a in args]
            arg_code_list = [self.function_signature(var, False) if isinstance(var, FunctionAddress)
                                else get_var_arg(arg, var) for arg, var in zip(args, var_list)]
            arg_code = ', '.join(arg_code_list)

        if self._additional_args :
            self._additional_args.pop()

        if isinstance(expr, FunctionAddress):
            return '{}(*{})({})'.format(ret_type, name, arg_code)
        else:
            return '{0}{1}({2})'.format(ret_type, name, arg_code)

    def _print_IndexedElement(self, expr):
        base = expr.base
        inds = list(expr.indices)
        base_shape = base.shape
        allow_negative_indexes = True if isinstance(base, PythonTuple) else base.allows_negative_indexes
        for i, ind in enumerate(inds):
            if isinstance(ind, PyccelUnarySub) and isinstance(ind.args[0], LiteralInteger):
                inds[i] = PyccelMinus(base_shape[i], ind.args[0], simplify = True)
            else:
                #indices of indexedElement of len==1 shouldn't be a tuple
                if isinstance(ind, tuple) and len(ind) == 1:
                    inds[i].args = ind[0]
                if allow_negative_indexes and \
                        not isinstance(ind, LiteralInteger) and not isinstance(ind, Slice):
                    inds[i] = IfTernaryOperator(PyccelLt(ind, LiteralInteger(0)),
                        PyccelAdd(base_shape[i], ind, simplify = True), ind)
        #set dtype to the C struct types
        dtype = self._print(expr.dtype)
        dtype = self.find_in_ndarray_type_registry(dtype, expr.precision)
        base_name = self._print(base)
        if getattr(base, 'is_ndarray', False) or isinstance(base, HomogeneousTupleVariable):
            if expr.rank > 0:
                #managing the Slice input
                for i , ind in enumerate(inds):
                    if isinstance(ind, Slice):
                        inds[i] = self._new_slice_with_processed_arguments(ind, PyccelArraySize(base, i),
                            allow_negative_indexes)
                    else:
                        inds[i] = Slice(ind, PyccelAdd(ind, LiteralInteger(1), simplify = True), LiteralInteger(1))
                inds = [self._print(i) for i in inds]
                return "array_slicing(%s, %s, %s)" % (base_name, expr.rank, ", ".join(inds))
            inds = [self._cast_to(i, NativeInteger(), 8).format(self._print(i)) for i in inds]
        else:
            raise NotImplementedError(expr)
        return "GET_ELEMENT(%s, %s, %s)" % (base_name, dtype, ", ".join(inds))


    def _cast_to(self, expr, dtype, precision):
        """ add cast to an expression when needed
        parameters
        ----------
            expr      : PyccelAstNode
                the expression to be cast
            dtype     : Datatype
                base type of the cast
            precision : integer
                precision of the base type of the cast

        Return
        ------
            String
                Return format string that contains the desired cast type
                NB: You should insert the expression to be cast in the string after using this function.
        """
        if (expr.dtype != dtype or expr.precision != precision):
            cast=self.find_in_dtype_registry(self._print(dtype), precision)
            return '({}){{}}'.format(cast)
        return '{}'

    def _print_DottedVariable(self, expr):
        """convert dotted Variable to their C equivalent"""

        name_code = self._print(expr.name)
        if self.stored_in_c_pointer(expr.lhs):
            code = f'{self._print(ObjectAddress(expr.lhs))}->{name_code}'
        else:
            lhs_code = self._print(expr.lhs)
            code = f'{lhs_code}.{name_code}'
        if self.stored_in_c_pointer(expr):
            return f'(*{code})'
        else:
            return code

    @staticmethod
    def _new_slice_with_processed_arguments(_slice, array_size, allow_negative_index):
        """ Create new slice with informations collected from old slice and decorators

        Parameters
        ----------
            _slice : Slice
                slice needed to collect (start, stop, step)
            array_size : PyccelArraySize
                call to function size()
            allow_negative_index : Bool
                True when the decorator allow_negative_index is present
        Returns
        -------
            Slice
        """
        start = LiteralInteger(0) if _slice.start is None else _slice.start
        stop = array_size if _slice.stop is None else _slice.stop

        # negative start and end in slice
        if isinstance(start, PyccelUnarySub) and isinstance(start.args[0], LiteralInteger):
            start = PyccelMinus(array_size, start.args[0], simplify = True)
        elif allow_negative_index and not isinstance(start, (LiteralInteger, PyccelArraySize)):
            start = IfTernaryOperator(PyccelLt(start, LiteralInteger(0)),
                            PyccelMinus(array_size, start, simplify = True), start)

        if isinstance(stop, PyccelUnarySub) and isinstance(stop.args[0], LiteralInteger):
            stop = PyccelMinus(array_size, stop.args[0], simplify = True)
        elif allow_negative_index and not isinstance(stop, (LiteralInteger, PyccelArraySize)):
            stop = IfTernaryOperator(PyccelLt(stop, LiteralInteger(0)),
                            PyccelMinus(array_size, stop, simplify = True), stop)

        # steps in slices
        step = _slice.step

        if step is None:
            step = LiteralInteger(1)

        # negative step in slice
        elif isinstance(step, PyccelUnarySub) and isinstance(step.args[0], LiteralInteger):
            start = PyccelMinus(array_size, LiteralInteger(1), simplify = True) if _slice.start is None else start
            stop = LiteralInteger(0) if _slice.stop is None else stop

        # variable step in slice
        elif allow_negative_index and step and not isinstance(step, LiteralInteger):
            og_start = start
            start = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)), start, PyccelMinus(stop, LiteralInteger(1), simplify = True))
            stop = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)), stop, og_start)

        return Slice(start, stop, step)

    def _print_NumpyArraySize(self, expr):
        arg = expr.arg
        if self.stored_in_c_pointer(arg):
            return '{}->length'.format(self._print(ObjectAddress(arg)))
        return '{}.length'.format(self._print(arg))

    def _print_PyccelArraySize(self, expr):
        arg    = expr.arg
        if self.stored_in_c_pointer(arg):
            return '{}->shape[{}]'.format(self._print(ObjectAddress(arg)), self._print(expr.index))
        return '{}.shape[{}]'.format(self._print(arg), self._print(expr.index))

    def _print_Allocate(self, expr):

        # Skip allocation
        if expr.variable.cls_base == ListClass or isinstance(expr.variable, PythonList):
            return ""

        # Code related to allocation of ndarrays
        free_code = ''
        #free the array if its already allocated and checking if its not null if the status is unknown
        if  (expr.status == 'unknown'):
            free_code = 'if (%s.shape != NULL)\n' % self._print(expr.variable.name)
            free_code += "{{\n{}}}\n".format(self._print(Deallocate(expr.variable)))
        elif  (expr.status == 'allocated'):
            free_code += self._print(Deallocate(expr.variable))
        self.add_import(c_imports['ndarrays'])
        shape = ", ".join(self._print(i) for i in expr.shape)
        dtype = self._print(expr.variable.dtype)
        dtype = self.find_in_ndarray_type_registry(dtype, expr.variable.precision)
        shape_dtype = self.find_in_dtype_registry('int', 8)
        shape_Assign = "("+ shape_dtype +"[]){" + shape + "}"
        is_view = 'false' if expr.variable.on_heap else 'true'
        alloc_code = "{} = array_create({}, {}, {}, {});\n".format(
                self._print(expr.variable), len(expr.shape), shape_Assign, dtype,
                is_view)
        return '{}{}'.format(free_code, alloc_code)

    def _print_Deallocate(self, expr):
        if isinstance(expr.variable, InhomogeneousTupleVariable):
            return ''.join(self._print(Deallocate(v)) for v in expr.variable)
        if expr.variable.cls_base == ListClass:
            return 'free_list({});\n'.format(self._print(ObjectAddress(expr.variable)))
        if expr.variable.is_alias:
            return 'free_pointer({});\n'.format(self._print(expr.variable))
        return 'free_array({});\n'.format(self._print(expr.variable))

    def _print_Slice(self, expr):
        start = self._print(expr.start)
        stop = self._print(expr.stop)
        step = self._print(expr.step)
        return 'new_slice({}, {}, {})'.format(start, stop, step)

    def _print_NumpyUfuncBase(self, expr):
        """ Convert a Python expression with a Numpy function call to C
        function call

        Parameters
        ----------
            expr : Pyccel ast node
                Python expression with a Numpy function call

        Returns
        -------
            string
                Equivalent expression in C language

        Example
        -------
            numpy.cos(x) ==> cos(x)

        """
        # add necessary include
        self.add_import(c_imports['math'])
        type_name = type(expr).__name__
        try:
            func_name = numpy_ufunc_to_c_float[type_name]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')
        args = []
        for arg in expr.args:
            if arg.dtype is NativeComplex():
                self.add_import(c_imports['complex'])
                try:
                    func_name = numpy_ufunc_to_c_complex[type_name]
                    args.append(self._print(arg))
                except KeyError:
                    errors.report(INCOMPATIBLE_TYPEVAR_TO_FUNC.format(type_name) ,severity='fatal')
            elif arg.dtype is not NativeFloat():
                args.append(self._print(NumpyFloat(arg)))
            else :
                args.append(self._print(arg))
        code_args = ', '.join(args)
        return '{0}({1})'.format(func_name, code_args)

    def _print_NumpySign(self, expr):
        """ Print the corresponding C function for a call to Numpy.sign

        Parameters
        ----------
            expr : Pyccel ast node
                Python expression with Numpy.sign call

        Returns
        -------
            string
                Equivalent internal function in C

        Example
        -------
            import numpy

            numpy.sign(x) => isign(x)   (x is integer)
            numpy.sign(x) => fsign(x)   (x if float)
            numpy.sign(x) => csign(x)   (x is complex)

        """
        self.add_import(c_imports['numpy_c'])
        dtype = expr.dtype
        func = ''
        if isinstance(dtype, NativeInteger):
            func = 'isign'
        elif isinstance(dtype, NativeFloat):
            func = 'fsign'
        elif isinstance(dtype, NativeComplex):
            func = 'csign'

        return f'{func}({self._print(expr.args[0])})'

    def _print_MathFunctionBase(self, expr):
        """ Convert a Python expression with a math function call to C
        function call

        Parameters
        ----------
            expr : Pyccel ast node
                Python expression with a Math function call

        Returns
        -------
            string
                Equivalent expression in C language

        ------
        Example:
        --------
            math.sin(x) ==> sin(x)

        """
        # add necessary include
        type_name = type(expr).__name__
        try:
            func_name = math_function_to_c[type_name]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')

        if func_name.startswith("pyc"):
            self.add_import(c_imports['pyc_math_c'])
        else:
            if expr.dtype is NativeComplex():
                self.add_import(c_imports['cmath'])
            else:
                self.add_import(c_imports['math'])
        args = []
        for arg in expr.args:
            if arg.dtype != NativeFloat() and not func_name.startswith("pyc"):
                args.append(self._print(NumpyFloat(arg)))
            else:
                args.append(self._print(arg))
        code_args = ', '.join(args)
        if expr.dtype == NativeInteger():
            cast_type = self.find_in_dtype_registry('int', expr.precision)
            return '({0}){1}({2})'.format(cast_type, func_name, code_args)
        return '{0}({1})'.format(func_name, code_args)

    def _print_MathIsfinite(self, expr):
        """Convert a Python expression with a math isfinite function call to C
        function call"""
        # add necessary include
        self.add_import(c_imports['math'])
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "isfinite({})".format(code_arg)

    def _print_MathIsinf(self, expr):
        """Convert a Python expression with a math isinf function call to C
        function call"""
        # add necessary include
        self.add_import(c_imports['math'])
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "isinf({})".format(code_arg)

    def _print_MathIsnan(self, expr):
        """Convert a Python expression with a math isnan function call to C
        function call"""
        # add necessary include
        self.add_import(c_imports['math'])
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "isnan({})".format(code_arg)

    def _print_MathTrunc(self, expr):
        """Convert a Python expression with a math trunc function call to C
        function call"""
        # add necessary include
        self.add_import(c_imports['math'])
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(NumpyFloat(arg))
        else:
            code_arg = self._print(arg)
        return "trunc({})".format(code_arg)

    def _print_FunctionAddress(self, expr):
        return expr.name

    def _print_NumpyWhere(self, expr):
        cond = self._print(expr.condition)
        value_true = self._print(expr.value_true)
        value_false = self._print(expr.value_false)
        stmt = '{cond} ? {true} : {false}'.format(cond = cond,
                true = value_true, false = value_false)
        return stmt

    def _print_Rand(self, expr):
        raise NotImplementedError("Rand not implemented")

    def _print_NumpyRandint(self, expr):
        raise NotImplementedError("Randint not implemented")

    def _print_NumpyMod(self, expr):
        return self._print(PyccelMod(*expr.args))

    def _print_NumpyLinspace(self, expr):
        template = '({start} + {index}*{step})'
        if not isinstance(expr.endpoint, LiteralFalse):
            template = '({start} + {index}*{step})'
            lhs_source = expr.get_user_nodes(Assign)[0].lhs
            lhs_source.substitute(expr.ind, PyccelMinus(expr.num, LiteralInteger(1), simplify = True))
            lhs = self._print(lhs_source)

            if isinstance(expr.endpoint, LiteralTrue):
                cond_template = lhs + ' = {stop}'
            else:
                cond_template = lhs + ' = {cond} ? {stop} : ' + lhs

        dtype = self._print(expr.dtype)
        v = self._cast_to(expr.stop, dtype, expr.precision).format(self._print(expr.stop))

        init_value = template.format(
            start = self._print(expr.start),
            step  = self._print(expr.step),
            index = self._print(expr.ind),
        )
        if isinstance(expr.endpoint, LiteralFalse):
            code = init_value
        elif isinstance(expr.endpoint, LiteralTrue):
            code = init_value + ';\n' + cond_template.format(stop = v)
        else:
            code = init_value + ';\n' + cond_template.format(cond=self._print(expr.endpoint),stop = v)

        return code

    def _print_Interface(self, expr):
        return ""

    def _print_FunctionDef(self, expr):
        if expr.is_inline:
            return ''
        self.set_scope(expr.scope)

        if len(expr.results) > 1:
            self._additional_args.append(expr.results)
        body  = self._print(expr.body)
        decs  = [Declare(i.dtype, i) if isinstance(i, Variable) else FuncAddressDeclare(i) for i in expr.local_vars]
        if len(expr.results) <= 1 :
            for i in expr.results:
                if isinstance(i, Variable) and not i.is_temp:
                    decs += [Declare(i.dtype, i)]
                elif not isinstance(i, Variable):
                    decs += [FuncAddressDeclare(i)]
        arguments = [a.var for a in expr.arguments]
        decs += [Declare(v.dtype,v) for v in self.scope.variables.values() \
                if v not in chain(expr.local_vars, expr.results, arguments)]
        decs  = ''.join(self._print(i) for i in decs)

        sep = self._print(SeparatorComment(40))
        if self._additional_args :
            self._additional_args.pop()
        for i in expr.imports:
            self.add_import(i)
        doc_string = self._print(expr.doc_string) if expr.doc_string else ''

        parts = [sep,
                 doc_string,
                '{signature}\n{{\n'.format(signature=self.function_signature(expr)),
                 decs,
                 body,
                 '}\n',
                 sep]

        self.exit_scope()

        return ''.join(p for p in parts if p)

    def stored_in_c_pointer(self, a):
        """
        Indicates whether the object a needs to be stored in a pointer
        in c code

        Parameters
        ----------
        a : PyccelAstNode
        """
        if isinstance(a, (Nil, ObjectAddress)):
            return True
        if isinstance(a, FunctionCall):
            results = a.funcdef.results
            if len(results)==1:
                a = a.funcdef.results[0]
            else:
                return False

        if not isinstance(a, Variable):
            return False
        return (a.is_alias and not a.is_ndarray) or a.is_optional or \
                any(a is bi for b in self._additional_args for bi in b)

    def _print_FunctionCall(self, expr):
        func = expr.funcdef
        if func.is_inline:
            return self._handle_inline_func_call(expr)
         # Ensure the correct syntax is used for pointers
        args = []
        if isinstance(expr, DottedFunctionCall):
            obj = expr.prefix
            if getattr(obj, 'cls_base', None) == ListClass:
                args.append(obj)
            elif isinstance(obj, PythonList):
                var = self.scope.get_temporary_variable(obj.dtype, rank=obj.rank, memory_handling='heap', cls_base=ListClass)
                self._additional_code += self._print(Assign(var, obj))
                args.append(var)

        for a, f in zip(expr.args, func.arguments):
            a = a.value if a else Nil()
            f = f.var
            if self.stored_in_c_pointer(f):
                if isinstance(a, Variable):
                    args.append(ObjectAddress(a))
                elif not self.stored_in_c_pointer(a):
                    tmp_var = self.scope.get_temporary_variable(f.dtype)
                    assign = Assign(tmp_var, a)
                    self._additional_code += self._print(assign)
                    args.append(ObjectAddress(tmp_var))
                else:
                    args.append(a)
            else :
                args.append(a)

        args += self._temporary_args
        self._temporary_args = []
        args = ', '.join(['{}'.format(self._print(a)) for a in args])

        call_code = f'{func.name}({args})'
        if not func.results:
            return f'{call_code};\n'
        else:
            return call_code

    def _print_Constant(self, expr):
        """ Convert a Python expression with a math constant call to C
        function call

        Parameters
        ----------
            expr : Pyccel ast node
                Python expression with a Math constant

        Returns
        -------
            string
                String represent the value of the constant

        Example
        -------
            math.pi ==> 3.14159265358979

        """
        val = LiteralFloat(expr.value)
        return self._print(val)

    def _print_Return(self, expr):
        code = ''
        args = [ObjectAddress(a) if isinstance(a, Variable) and self.stored_in_c_pointer(a) else a for a in expr.expr]

        if len(args) == 0:
            return 'return;\n'

        if len(args) > 1:
            if expr.stmt:
                return self._print(expr.stmt)+'return 0;\n'
            return 'return 0;\n'

        if expr.stmt:
            # get Assign nodes from the CodeBlock object expr.stmt.
            last_assign = expr.stmt.get_attribute_nodes(Assign, excluded_nodes=FunctionCall)
            deallocate_nodes = expr.stmt.get_attribute_nodes(Deallocate, excluded_nodes=(Assign,))
            vars_in_deallocate_nodes = [i.variable for i in deallocate_nodes]

            # Check the Assign objects list in case of
            # the user assigns a variable to an object contains IndexedElement object.
            if not last_assign:
                code = ''+self._print(expr.stmt)
            elif isinstance(last_assign[-1], AugAssign):
                last_assign[-1].lhs.is_temp = False
                code = ''+self._print(expr.stmt)
            else:
                # make sure that stmt contains one assign node.
                last_assign = last_assign[-1]
                variables = last_assign.rhs.get_attribute_nodes(Variable)
                unneeded_var = not any(b in vars_in_deallocate_nodes or b.is_ndarray for b in variables)
                if unneeded_var:
                    code = ''.join(self._print(a) for a in expr.stmt.body if a is not last_assign)
                    return code + 'return {};\n'.format(self._print(last_assign.rhs))
                else:
                    last_assign.lhs.is_temp = False
                    code = self._print(expr.stmt)

        return code + 'return {0};\n'.format(self._print(args[0]))

    def _print_Pass(self, expr):
        return '// pass\n'

    def _print_Nil(self, expr):
        return 'NULL'

    def _print_NilArgument(self, expr):
        raise errors.report("Trying to use optional argument in inline function without providing a variable",
                symbol=expr,
                severity='fatal')

    def _print_PyccelAdd(self, expr):
        return ' + '.join(self._print(a) for a in expr.args)

    def _print_PyccelMinus(self, expr):
        args = [self._print(a) for a in expr.args]
        if len(args) == 1:
            return '-{}'.format(args[0])
        return ' - '.join(args)

    def _print_PyccelMul(self, expr):
        return ' * '.join(self._print(a) for a in expr.args)

    def _print_PyccelDiv(self, expr):
        if all(a.dtype is NativeInteger() for a in expr.args):
            args = [NumpyFloat(a) for a in expr.args]
        else:
            args = expr.args
        return  ' / '.join(self._print(a) for a in args)

    def _print_PyccelFloorDiv(self, expr):
        self.add_import(c_imports['math'])
        # the result type of the floor division is dependent on the arguments
        # type, if all arguments are integers the result is integer otherwise
        # the result type is float
        need_to_cast = all(a.dtype is NativeInteger() for a in expr.args)
        code = ' / '.join(self._print(a if a.dtype is NativeFloat() else NumpyFloat(a)) for a in expr.args)
        if (need_to_cast):
            cast_type = self.find_in_dtype_registry('int', expr.precision)
            return "({})floor({})".format(cast_type, code)
        return "floor({})".format(code)

    def _print_PyccelRShift(self, expr):
        return ' >> '.join(self._print(a) for a in expr.args)

    def _print_PyccelLShift(self, expr):
        return ' << '.join(self._print(a) for a in expr.args)

    def _print_PyccelBitXor(self, expr):
        if expr.dtype is NativeBool():
            return '{0} != {1}'.format(self._print(expr.args[0]), self._print(expr.args[1]))
        return ' ^ '.join(self._print(a) for a in expr.args)

    def _print_PyccelBitOr(self, expr):
        if expr.dtype is NativeBool():
            return ' || '.join(self._print(a) for a in expr.args)
        return ' | '.join(self._print(a) for a in expr.args)

    def _print_PyccelBitAnd(self, expr):
        if expr.dtype is NativeBool():
            return ' && '.join(self._print(a) for a in expr.args)
        return ' & '.join(self._print(a) for a in expr.args)

    def _print_PyccelInvert(self, expr):
        return '~{}'.format(self._print(expr.args[0]))

    def _print_PyccelAssociativeParenthesis(self, expr):
        return '({})'.format(self._print(expr.args[0]))

    def _print_PyccelUnary(self, expr):
        return '+{}'.format(self._print(expr.args[0]))

    def _print_PyccelUnarySub(self, expr):
        return '-{}'.format(self._print(expr.args[0]))

    def _print_AugAssign(self, expr):
        op = expr.op
        lhs = expr.lhs
        rhs = expr.rhs

        if op == '%' and isinstance(lhs.dtype, NativeFloat):
            _expr = expr.to_basic_assign()
            return self._print(_expr)

        lhs_code = self._print(lhs)
        rhs_code = self._print(rhs)
        return f'{lhs_code} {op}= {rhs_code};\n'

    def _print_Assign(self, expr):
        prefix_code = ''
        lhs = expr.lhs
        rhs = expr.rhs
        if isinstance(lhs, Variable) and lhs.is_optional:
            if lhs in self._optional_partners:
                # Collect temporary variable which provides
                # allocated memory space for this optional variable
                tmp_var = self._optional_partners[lhs]
            else:
                # Create temporary variable to provide allocated
                # memory space before assigning to the pointer value
                # (may be NULL)
                tmp_var = self.scope.get_temporary_variable(lhs,
                        is_optional = False)
                self._optional_partners[lhs] = tmp_var
            # Point optional variable at an allocated memory space
            prefix_code = self._print(AliasAssign(lhs, tmp_var))
        if isinstance(rhs, FunctionCall) and isinstance(rhs.dtype, NativeTuple):
            self._temporary_args = [ObjectAddress(a) for a in lhs]
            return prefix_code+'{};\n'.format(self._print(rhs))
        # Inhomogenous tuples are unravelled and therefore do not exist in the c printer
        if isinstance(rhs, (NumpyArray, PythonTuple)):
            return prefix_code+self.copy_NumpyArray_Data(expr)
        if isinstance(rhs, (NumpyFull)):
            return prefix_code+self.arrayFill(expr)
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return prefix_code+'{} = {};\n'.format(lhs, rhs)

    def _print_AliasAssign(self, expr):
        lhs_var = expr.lhs
        rhs_var = expr.rhs

        lhs_address = ObjectAddress(lhs_var)
        rhs_address = ObjectAddress(rhs_var)

        # the below condition handles the case of reassinging a pointer to an array view.
        # setting the pointer's is_view attribute to false so it can be ignored by the free_pointer function.
        if not self.stored_in_c_pointer(lhs_var) and \
                isinstance(lhs_var, Variable) and lhs_var.is_ndarray:
            rhs = self._print(rhs_var)

            if isinstance(rhs_var, Variable) and rhs_var.is_ndarray:
                lhs = self._print(lhs_address)
                if lhs_var.order == rhs_var.order:
                    return 'alias_assign({}, {});\n'.format(lhs, rhs)
                else:
                    return 'transpose_alias_assign({}, {});\n'.format(lhs, rhs)
            else:
                lhs = self._print(lhs_var)
                return '{} = {};\n'.format(lhs, rhs)
        else:
            lhs = self._print(lhs_address)
            rhs = self._print(rhs_address)

            return '{} = {};\n'.format(lhs, rhs)

    def _print_For(self, expr):
        self.set_scope(expr.scope)

        indices = expr.iterable.loop_counters
        index = indices[0] if indices else expr.target
        if expr.iterable.num_loop_counters_required:
            self.scope.insert_variable(index)

        target   = index
        iterable = expr.iterable.get_range()

        if not isinstance(iterable, PythonRange):
            # Only iterable currently supported is PythonRange
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')

        counter    = self._print(target)
        body       = self._print(expr.body)

        additional_assign = CodeBlock(expr.iterable.get_assigns(expr.target))
        body = self._print(additional_assign) + body

        start = self._print(iterable.start)
        stop  = self._print(iterable.stop )
        step  = self._print(iterable.step )

        test_step = iterable.step
        if isinstance(test_step, PyccelUnarySub):
            test_step = iterable.step.args[0]

        self.exit_scope()
        # testing if the step is a value or an expression
        if isinstance(test_step, Literal):
            op = '>' if isinstance(iterable.step, PyccelUnarySub) else '<'
            return ('for ({counter} = {start}; {counter} {op} {stop}; {counter} += '
                        '{step})\n{{\n{body}}}\n').format(counter=counter, start=start, op=op,
                                                          stop=stop, step=step, body=body)
        else:
            return (
                'for ({counter} = {start}; ({step} > 0) ? ({counter} < {stop}) : ({counter} > {stop}); {counter} += '
                '{step})\n{{\n{body}}}\n').format(counter=counter, start=start,
                                                  stop=stop, step=step, body=body)

    def _print_FunctionalFor(self, expr):
        loops = ''.join(self._print(i) for i in expr.loops)
        return loops

    def _print_CodeBlock(self, expr):
        if not expr.unravelled:
            body_exprs = expand_to_loops(expr,
                    self.scope.get_temporary_variable, self.scope,
                    language_has_vectors = False)
        else:
            body_exprs = expr.body
        body_stmts = []
        for b in body_exprs :
            code = self._print(b)
            code = self._additional_code + code
            self._additional_code = ''
            body_stmts.append(code)
        return ''.join(self._print(b) for b in body_stmts)

    def _print_Idx(self, expr):
        return self._print(expr.label)

    def _print_Exp1(self, expr):
        return "M_E"

    def _print_Pi(self, expr):
        return 'M_PI'

    def _print_Infinity(self, expr):
        return 'HUGE_VAL'

    def _print_NegativeInfinity(self, expr):
        return '-HUGE_VAL'

    def _print_PythonReal(self, expr):
        return 'creal({})'.format(self._print(expr.internal_var))

    def _print_PythonImag(self, expr):
        return 'cimag({})'.format(self._print(expr.internal_var))

    def _print_PythonConjugate(self, expr):
        return 'conj({})'.format(self._print(expr.internal_var))

    def _handle_is_operator(self, Op, expr):

        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        a = expr.args[0]
        b = expr.args[1]

        if Nil() in expr.args:
            lhs = ObjectAddress(expr.lhs) if isinstance(expr.lhs, Variable) else expr.lhs
            rhs = ObjectAddress(expr.rhs) if isinstance(expr.rhs, Variable) else expr.rhs

            lhs = self._print(lhs)
            rhs = self._print(rhs)
            return '{} {} {}'.format(lhs, Op, rhs)

        if (a.dtype is NativeBool() and b.dtype is NativeBool()):
            return '{} {} {}'.format(lhs, Op, rhs)
        else:
            errors.report(PYCCEL_RESTRICTION_IS_ISNOT,
                          symbol=expr, severity='fatal')

    def _print_PyccelIsNot(self, expr):
        return self._handle_is_operator("!=", expr)

    def _print_PyccelIs(self, expr):
        return self._handle_is_operator("==", expr)

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond is not True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if expr.has(Assign):
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s) {\n" % self._print(c))
                elif i == len(expr.args) - 1 and c is True:
                    lines.append("else {\n")
                else:
                    lines.append("else if (%s) {\n" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                lines.append("}\n")
            return "".join(lines)
        else:
            # The piecewise was used in an expression, need to do inline
            # operators. This has the downside that inline operators will
            # not work for statements that span multiple lines (Matrix or
            # Indexed expressions).
            ecpairs = ["((%s) ? (\n%s\n)\n" % (self._print(c), self._print(e))
                    for e, c in expr.args[:-1]]
            last_line = ": (\n%s\n)" % self._print(expr.args[-1].expr)
            return ": ".join(ecpairs) + last_line + " ".join([")"*len(ecpairs)])

    def _print_Constant(self, expr):
        if expr == math_constants['inf']:
            self.add_import(c_imports['math'])
            return 'HUGE_VAL'
        elif expr == math_constants['pi']:
            self.add_import(c_imports['math'])
            return 'M_PI'
        elif expr == math_constants['e']:
            self.add_import(c_imports['math'])
            return 'M_E'
        else:
            raise NotImplementedError("Constant not implemented")

    def _print_Variable(self, expr):
        if self.stored_in_c_pointer(expr):
            return '(*{0})'.format(expr.name)
        else:
            return expr.name

    def _print_FunctionDefArgument(self, expr):
        return self._print(expr.name)

    def _print_FunctionCallArgument(self, expr):
        return self._print(expr.value)

    def _print_ObjectAddress(self, expr):
        obj_code = self._print(expr.obj)
        if isinstance(expr.obj, ObjectAddress) or not self.stored_in_c_pointer(expr.obj):
            return f'&{obj_code}'
        else:
            if obj_code.startswith('(*') and obj_code.endswith(')'):
                return f'{obj_code[2:-1]}'
            else:
                return obj_code

    def _print_Comment(self, expr):
        comments = self._print(expr.text)

        return '/*' + comments + '*/\n'

    def _print_Assert(self, expr):
        condition = self._print(expr.test)
        self.add_import(c_imports['assert'])
        return "assert({0});\n".format(condition)

    def _print_PyccelSymbol(self, expr):
        return expr

    def _print_CommentBlock(self, expr):
        txts = expr.comments
        header = expr.header
        header_size = len(expr.header)

        ln = max(len(i) for i in txts)
        if ln<max(20, header_size+4):
            ln = 20
        top  = '/*' + '_'*int((ln-header_size)/2) + header + '_'*int((ln-header_size)/2) + '*/\n'
        ln = len(top)-4
        bottom = '/*' + '_'*ln + '*/\n'

        txts = ['/*' + t + ' '*(ln - len(t)) + '*/\n' for t in txts]

        body = ''.join(i for i in txts)

        return ''.join([top, body, bottom])

    def _print_EmptyNode(self, expr):
        return ''

    #=================== OMP ==================

    def _print_OmpAnnotatedComment(self, expr):
        clauses = ''
        if expr.combined:
            clauses = ' ' + expr.combined
        clauses += str(expr.txt)
        if expr.has_nowait:
            clauses = clauses + ' nowait'
        omp_expr = '#pragma omp {}{}\n'.format(expr.name, clauses)

        if expr.is_multiline:
            if expr.combined is None:
                omp_expr += '{\n'
            elif (expr.combined and "for" not in expr.combined):
                if ("masked taskloop" not in expr.combined) and ("distribute" not in expr.combined):
                    omp_expr += '{\n'

        return omp_expr

    def _print_Omp_End_Clause(self, expr):
        return '}\n'
    #=====================================

    def _print_Program(self, expr):
        self.set_scope(expr.scope)
        body  = self._print(expr.body)
        variables = self.scope.variables.values()
        decs = ''.join(self._print(Declare(v.dtype, v)) for v in variables)

        imports = [*expr.imports, *self._additional_imports.values()]
        imports = ''.join(self._print(i) for i in imports)

        self.exit_scope()
        return ('{imports}'
                'int main()\n{{\n'
                '{decs}'
                '{body}'
                'return 0;\n'
                '}}').format(imports=imports,
                                    decs=decs,
                                    body=body)

    #=================== MACROS ==================

    def _print_MacroShape(self, expr):
        var = expr.argument
        if not isinstance(var, (Variable, IndexedElement)):
            raise TypeError('Expecting a variable, given {}'.format(type(var)))
        shape = var.shape

        if len(shape) == 1:
            shape = shape[0]


        elif not(expr.index is None):
            if expr.index < len(shape):
                shape = shape[expr.index]
            else:
                shape = '1'

        return self._print(shape)

    def _print_MacroCount(self, expr):

        var = expr.argument

        if var.rank == 0:
            return '1'
        else:
            return self._print(functools.reduce(
                lambda x,y: PyccelMul(x,y,simplify=True), var.shape))

    def _print_PrecomputedCode(self, expr):
        return expr.code


    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = " "*self._default_settings["tabwidth"]
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')', '\n}')

        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(any(map(line.endswith, inc_token))) for line in code ]
        decrease = [ int(any(map(line.startswith, dec_token)))
                     for line in code ]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line == '' or line == '\n':
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        return pretty

def ccode(expr, filename, assign_to=None, **settings):
    """Converts an expr to a string of c code

    expr : Expr
        A pyccel expression to be converted.
    filename : str
        The name of the file being translated. Used in error printing
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
    return CCodePrinter(filename, **settings).doprint(expr, assign_to)
