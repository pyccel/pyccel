# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=R0201
# pylint: disable=missing-function-docstring
import functools
import operator

from sympy.core           import Tuple
from pyccel.ast.builtins  import PythonRange, PythonFloat, PythonComplex

from pyccel.ast.core      import Declare, IndexedVariable, Slice, ValuedVariable
from pyccel.ast.core      import FuncAddressDeclare, FunctionCall
from pyccel.ast.core      import Deallocate
from pyccel.ast.core      import FunctionAddress, PyccelArraySize
from pyccel.ast.core      import Nil, IfTernaryOperator
from pyccel.ast.core      import Assign, datatype, Variable, Import
from pyccel.ast.core      import SeparatorComment, VariableAddress
from pyccel.ast.core      import DottedName
from pyccel.ast.core      import create_incremented_string

from pyccel.ast.operators import PyccelAdd, PyccelMul, PyccelMinus, PyccelLt, PyccelGt
from pyccel.ast.operators import PyccelAssociativeParenthesis
from pyccel.ast.operators import PyccelUnarySub, PyccelLt

from pyccel.ast.datatypes import default_precision, str_dtype
from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeComplex, NativeReal, NativeTuple

from pyccel.ast.literals  import LiteralTrue, LiteralImaginaryUnit, LiteralFloat
from pyccel.ast.literals  import LiteralString, LiteralInteger, Literal

from pyccel.ast.numpyext import NumpyFull, NumpyArray
from pyccel.ast.numpyext import NumpyReal, NumpyImag, NumpyFloat


from pyccel.codegen.printing.codeprinter import CodePrinter

from pyccel.errors.errors   import Errors
from pyccel.errors.messages import (PYCCEL_RESTRICTION_TODO, INCOMPATIBLE_TYPEVAR_TO_FUNC,
                                    PYCCEL_RESTRICTION_IS_ISNOT )

from .fcode import python_builtin_datatypes

errors = Errors()

# TODO: add examples

__all__ = ["CCodePrinter", "ccode"]

# dictionary mapping sympy function to (argument_conditions, C_function).
# Used in CCodePrinter._print_Function(self)
known_functions = {
    "Abs": [(lambda x: not x.is_integer, "fabs")],
    "gamma": "tgamma",
    "sin"  : "sin",
    "cos"  : "cos",
    "tan"  : "tan",
    "asin" : "asin",
    "acos" : "acos",
    "atan" : "atan",
    "atan2": "atan2",
    "exp"  : "exp",
    "log"  : "log",
    "erf"  : "erf",
    "sinh" : "sinh",
    "cosh" : "cosh",
    "tanh" : "tanh",
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "floor": "floor",
    "ceiling": "ceil",
}

# dictionary mapping numpy function to (argument_conditions, C_function).
# Used in CCodePrinter._print_NumpyUfuncBase(self, expr)
numpy_ufunc_to_c_real = {
    'NumpyAbs'  : 'fabs',
    'NumpyFabs'  : 'fabs',
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

dtype_registry = {('real',8)    : 'double',
                  ('real',4)    : 'float',
                  ('complex',8) : 'double complex',
                  ('complex',4) : 'float complex',
                  ('int',4)     : 'int32_t',
                  ('int',8)     : 'int64_t',
                  ('int',2)     : 'int16_t',
                  ('int',1)     : 'int8_t',
                  ('bool',4)    : 'bool'}

ndarray_type_registry = {('real',8)    : 'nd_double',
                  ('real',4)    : 'nd_float',
                  ('complex',8) : 'nd_cdouble',
                  ('complex',4) : 'nd_cfloat',
                  ('int',8)     : 'nd_int64',
                  ('int',4)     : 'nd_int32',
                  ('int',2)     : 'nd_int16',
                  ('int',1)     : 'nd_int8',
                  ('bool',4)    : 'nd_bool'}

import_dict = {'omp_lib' : 'omp' }

class CCodePrinter(CodePrinter):
    """A printer to convert python expressions to strings of c code"""
    printmethod = "_ccode"
    language = "C"

    _default_settings = {
        'order': None,
        'full_prec': 'auto',
        'human': True,
        'precision': 15,
        'user_functions': {},
        'dereference': set()
    }

    def __init__(self, parser, settings=None):

        if parser.filename:
            errors.set_target(parser.filename, 'file')

        prefix_module = None if settings is None else settings.pop('prefix_module', None)
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = {} if settings is None else settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        self._dereference = set([] if settings is None else settings.get('dereference', []))
        self.prefix_module = prefix_module
        self._additional_imports = set(['stdlib'])
        self._parser = parser
        self._additional_code = ''
        self._additional_declare = []
        self._additional_args = []
        self._temporary_args = []

    def get_additional_imports(self):
        """return the additional imports collected in printing stage"""
        return self._additional_imports

    def _get_statement(self, codestring):
        return "%s;" % codestring

    def _get_comment(self, text):
        return "// {0}".format(text)

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    # ============ Elements ============ #

    def _print_PythonFloat(self, expr):
        value = self._print(expr.arg)
        type_name = self.find_in_dtype_registry('real', default_precision['real'])
        return '({0})({1})'.format(type_name, value)

    def _print_PythonInt(self, expr):
        value = self._print(expr.arg)
        type_name = self.find_in_dtype_registry('int', default_precision['int'])
        return '({0})({1})'.format(type_name, value)

    def _print_PythonBool(self, expr):
        value = self._print(expr.arg)
        return '({} != 0)'.format(value)

    def _print_LiteralInteger(self, expr):
        return str(expr.p)

    def _print_LiteralComplex(self, expr):
        if expr.real == LiteralFloat(0):
            return self._print(PyccelAssociativeParenthesis(PyccelMul(expr.imag, LiteralImaginaryUnit())))
        else:
            return self._print(PyccelAssociativeParenthesis(PyccelAdd(expr.real,
                            PyccelMul(expr.imag, LiteralImaginaryUnit()))))

    def _print_PythonComplex(self, expr):
        self._additional_imports.add("complex")
        if expr.is_cast:
            return self._print(expr.internal_var)
        else:
            return self._print(PyccelAssociativeParenthesis(PyccelAdd(expr.real,
                            PyccelMul(expr.imag, LiteralImaginaryUnit()))))

    def _print_LiteralImaginaryUnit(self, expr):
        return '_Complex_I'

    def _print_ModuleHeader(self, expr):
        name = expr.module.name
        # TODO: Add classes and interfaces
        funcs = '\n\n'.join('{};'.format(self.function_signature(f)) for f in expr.module.funcs)

        # Print imports last to be sure that all additional_imports have been collected
        imports = [*expr.module.imports, *map(Import, self._additional_imports)]
        imports = '\n'.join(self._print(i) for i in imports)

        return ('#ifndef {name}_H\n'
                '#define {name}_H\n\n'
                '{imports}\n\n'
                #'{classes}\n\n'
                '{funcs}\n\n'
                #'{interfaces}\n\n'
                '#endif // {name}_H').format(
                        name    = name.upper(),
                        imports = imports,
                        funcs   = funcs)

    def _print_Module(self, expr):
        body    = '\n\n'.join(self._print(i) for i in expr.body)

        # Print imports last to be sure that all additional_imports have been collected
        imports = [Import(expr.name), *map(Import, self._additional_imports)]
        imports = '\n'.join(self._print(i) for i in imports)
        return ('{imports}\n\n'
                '{body}').format(
                        imports = imports,
                        body    = body)

    def _print_Break(self, expr):
        return 'break;'

    def _print_Continue(self, expr):
        return 'continue;'

    def _print_While(self, expr):
        body = self._print(expr.body)
        cond = self._print(expr.test)
        return 'while({condi})\n{{\n{body}\n}}'.format(condi = cond, body = body)

    def _print_If(self, expr):
        lines = []
        for i, (c, e) in enumerate(expr.args):
            var = self._print(e)
            if (var == ''):
                break
            if i == 0:
                lines.append("if (%s)\n{" % self._print(c))
            elif i == len(expr.args) - 1 and c is LiteralTrue():
                lines.append("else\n{")
            else:
                lines.append("else if (%s)\n{" % self._print(c))
            lines.append("%s\n}" % var)
        return "\n".join(lines)

    def _print_IfTernaryOperator(self, expr):
        cond = self._print(expr.cond)
        value_true = self._print(expr.value_true)
        value_false = self._print(expr.value_false)
        return '({cond}) ? {true} : {false}'.format(cond = cond, true =value_true, false = value_false)

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
        self._additional_imports.add("math")

        first = self._print(expr.args[0])
        second = self._print(expr.args[1])

        if expr.dtype is NativeInteger():
            return "{} % {}".format(first, second)

        if expr.args[0].dtype is NativeInteger():
            first = self._print(PythonFloat(expr.args[0]))
        if expr.args[1].dtype is NativeInteger():
            second = self._print(PythonFloat(expr.args[1]))
        return "fmod({}, {})".format(first, second)

    def _print_PyccelPow(self, expr):
        b = expr.args[0]
        e = expr.args[1]

        if expr.dtype is NativeComplex():
            b = self._print(b if b.dtype is NativeComplex() else PythonComplex(b))
            e = self._print(e if e.dtype is NativeComplex() else PythonComplex(e))
            self._additional_imports.add("complex")
            return 'cpow({}, {})'.format(b, e)

        self._additional_imports.add("math")
        b = self._print(b if b.dtype is NativeReal() else PythonFloat(b))
        e = self._print(e if e.dtype is NativeReal() else PythonFloat(e))
        code = 'pow({}, {})'.format(b, e)
        if expr.dtype is NativeInteger():
            dtype = self._print(expr.dtype)
            prec  = expr.precision
            cast_type = self.find_in_dtype_registry(dtype, prec)
            return '({}){}'.format(cast_type, code)
        return code

    def _print_Import(self, expr):
        if isinstance(expr.source, DottedName):
            source = expr.source.name[-1]
        else:
            source = self._print(expr.source)

        # Get with a default value is not used here as it is
        # slower and on most occasions the import will not be in the
        # dictionary
        if source in import_dict: # pylint: disable=consider-using-get
            source = import_dict[source]

        if source is None:
            return ''
        else:
            return '#include <{0}.h>'.format(source)

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
        type_to_format = {('real',8)    : '%.12lf',
                          ('real',4)    : '%.12f',
                          ('complex',8) : '(%.12lf + %.12lfj)',
                          ('complex',4) : '(%.12f + %.12fj)',
                          ('int',4)     : '%d',
                          ('int',8)     : '%ld',
                          ('int',2)     : '%hd',
                          ('int',1)     : '%c',
                          ('bool',4)    : '%s',
                          ('string', 0) : '%s'}
        try:
            arg_format = type_to_format[(self._print(var.dtype), var.precision)]
        except KeyError:
            errors.report("{} type is not supported currently".format(var.dtype), severity='fatal')
        if var.dtype is NativeComplex():
            arg = '{}, {}'.format(self._print(NumpyReal(var)), self._print(NumpyImag(var)))
        elif var.dtype is NativeBool():
            arg = '{} ? "True" : "False"'.format(self._print(var))
        else:
            arg = self._print(var)
        return arg_format, arg

    def extract_function_call_results(self, expr):
        tmp_list = [self.create_tmp_var(a) for a in expr.funcdef.results]
        return tmp_list


    def _print_PythonPrint(self, expr):
        self._additional_imports.add("stdio")
        args_format = []
        args = []
        end = '\n'
        sep = ' '
        for f in expr.expr:
            if isinstance(f, ValuedVariable):
                if f.name == 'sep'      :   sep = str(f.value)
                elif f.name == 'end'    :   end = str(f.value)
            elif isinstance(f, FunctionCall) and isinstance(f.dtype, NativeTuple):
                tmp_list = self.extract_function_call_results(f)
                tmp_arg_format_list = []
                for a in tmp_list:
                    arg_format, arg = self.get_print_format_and_arg(a)
                    tmp_arg_format_list.append(arg_format)
                    args.append(arg)
                args_format.append('({})'.format(', '.join(tmp_arg_format_list)))
                assign = Assign(tmp_list, f)
                self._additional_code += self._print(assign) + '\n'
            else:
                arg_format, arg = self.get_print_format_and_arg(f)
                args_format.append(arg_format)
                args.append(arg)
        args_format = sep.join(args_format)
        args_format += end
        args_format = self._print(LiteralString(args_format))
        code = ', '.join([args_format, *args])
        return "printf({});".format(code)

    def find_in_dtype_registry(self, dtype, prec):
        try :
            return dtype_registry[(dtype, prec)]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO,
                    symbol = "{}[kind = {}]".format(dtype, prec),
                    severity='fatal')

    def find_in_ndarray_type_registry(self, dtype, prec):
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
            self._additional_imports.add('stdint')
        dtype = self.find_in_dtype_registry(dtype, prec)
        if rank > 0:
            if expr.is_ndarray:
                return 't_ndarray '
            errors.report(PYCCEL_RESTRICTION_TODO, symbol="rank > 0",severity='fatal')

        if self.stored_in_c_pointer(expr):
            return '{0} *'.format(dtype)
        else:
            return '{0} '.format(dtype)

    def _print_FuncAddressDeclare(self, expr):
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
            # TODO: extract informations needed for printing in case of function argument which itself has a function argument
            arg_code = ', '.join('{}'.format(self._print_FuncAddressDeclare(i))
                        if isinstance(i, FunctionAddress) else '{0}{1}'.format(self.get_declare_type(i), i)
                        for i in args)
        return '{}(*{})({});'.format(ret_type, name, arg_code)

    def _print_Declare(self, expr):
        declaration_type = self.get_declare_type(expr.variable)
        variable = self._print(expr.variable.name)

        return '{0}{1};'.format(declaration_type, variable)

    def _print_NativeBool(self, expr):
        self._additional_imports.add('stdbool')
        return 'bool'

    def _print_NativeInteger(self, expr):
        return 'int'

    def _print_NativeReal(self, expr):
        return 'real'

    def _print_NativeVoid(self, expr):
        return 'void'

    def _print_NativeComplex(self, expr):
        self._additional_imports.add('complex')
        return 'complex'
    def _print_NativeString(self, expr):
        return 'string'

    def function_signature(self, expr):
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
            arg_code = ', '.join('{}'.format(self.function_signature(i))
                        if isinstance(i, FunctionAddress) else '{0}{1}'.format(self.get_declare_type(i), i)
                        for i in args)
        if isinstance(expr, FunctionAddress):
            return '{}(*{})({})'.format(ret_type, name, arg_code)
        else:
            return '{0}{1}({2})'.format(ret_type, name, arg_code)

    def _print_IndexedElement(self, expr):
        if isinstance(expr.base, IndexedVariable):
            base = expr.base.internal_variable
        else:
            base = expr.base
        inds = list(expr.indices)
        inds = inds[::-1]
        base_shape = base.shape
        allow_negative_indexes = (isinstance(expr.base, IndexedVariable) and \
                base.allows_negative_indexes)
        for i, ind in enumerate(inds):
            if isinstance(ind, PyccelUnarySub) and isinstance(ind.args[0], LiteralInteger):
                inds[i] = PyccelMinus(base_shape[i], ind.args[0])
            else:
                #indices of indexedElement of len==1 shouldn't be a Tuple
                if isinstance(ind, Tuple) and len(ind) == 1:
                    inds[i].args = ind[0]
                if allow_negative_indexes and \
                        not isinstance(ind, LiteralInteger) and not isinstance(ind, Slice):
                    inds[i] = IfTernaryOperator(PyccelLt(ind, LiteralInteger(0)),
                        PyccelAdd(base_shape[i], ind), ind)
        #set dtype to the C struct types
        dtype = self._print(expr.dtype)
        dtype = self.find_in_ndarray_type_registry(dtype, expr.precision)
        base_name = self._print(base.name)
        if base.is_ndarray:
            if expr.rank > 0:
                #managing the Slice input
                for i , ind in enumerate(inds):
                    if isinstance(ind, Slice):
                        inds[i] = self._new_slice_with_processed_arguments(ind, PyccelArraySize(base, i),
                            allow_negative_indexes)
                inds = [self._print(i) for i in inds]
                return "array_slicing(%s, %s)" % (base_name, ", ".join(inds))
            inds = [self._print(i) for i in inds]
        else:
            raise NotImplementedError(expr)
        return "%s.%s[get_index(%s, %s)]" % (base_name, dtype, base_name, ", ".join(inds))

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
            start = PyccelMinus(array_size, start.args[0])
        elif allow_negative_index and not isinstance(start, LiteralInteger):
            start = IfTernaryOperator(PyccelLt(start, LiteralInteger(0)),
                            PyccelMinus(array_size, start), start)

        if isinstance(stop, PyccelUnarySub) and isinstance(stop.args[0], LiteralInteger):
            stop = PyccelMinus(array_size, stop.args[0])
        elif allow_negative_index and not isinstance(stop, LiteralInteger):
            stop = IfTernaryOperator(PyccelLt(stop, LiteralInteger(0)),
                            PyccelMinus(array_size, stop), stop)

        # steps in slices
        step = _slice.step

        if step is None:
            step = LiteralInteger(1)

        # negative step in slice
        elif isinstance(step, PyccelUnarySub) and isinstance(step.args[0], LiteralInteger):
            start = array_size if _slice.start is None else start
            stop = LiteralInteger(0) if _slice.stop is None else stop

        # variable step in slice
        elif allow_negative_index and step and not isinstance(step, LiteralInteger):
            start = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)), start, stop)
            stop = IfTernaryOperator(PyccelGt(step, LiteralInteger(0)), stop, start)

        return Slice(start, stop, step)

    def _print_PyccelArraySize(self, expr):
        return '{}.shape[{}]'.format(expr.arg, expr.index)

    def _print_Allocate(self, expr):
        free_code = ''
        #free the array if its already allocated and checking if its not null if the status is unknown
        if  (expr.status == 'unknown'):
            free_code = 'if (%s.shape != NULL)\n' % self._print(expr.variable.name)
            free_code += "{{\n{};\n}}\n".format(self._print(Deallocate(expr.variable)))
        elif  (expr.status == 'allocated'):
            free_code += self._print(Deallocate(expr.variable))
        self._additional_imports.add('ndarrays')
        shape = expr.shape
        shape = [self._print(i) for i in shape]
        shape = ", ".join(a for a in shape)
        dtype = self._print(expr.variable.dtype)
        dtype = self.find_in_ndarray_type_registry(dtype, expr.variable.precision)
        shape_dtype = self.find_in_dtype_registry('int', 4)
        shape_Assign = "("+ shape_dtype +"[]){" + shape + "}"
        alloc_code = "{} = array_create({}, {}, {});".format(expr.variable, len(expr.shape), shape_Assign, dtype)
        return '{}\n{}'.format(free_code, alloc_code)

    def _print_Deallocate(self, expr):
        if expr.variable.is_pointer:
            return 'free_pointer({});'.format(self._print(expr.variable))
        return 'free_array({});'.format(self._print(expr.variable))

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
        self._additional_imports.add('math')
        type_name = type(expr).__name__
        try:
            func_name = numpy_ufunc_to_c_real[type_name]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')
        args = []
        for arg in expr.args:
            if arg.dtype is NativeComplex():
                self._additional_imports.add('complex')
                try:
                    func_name = numpy_ufunc_to_c_complex[type_name]
                    args.append(self._print(arg))
                except KeyError:
                    errors.report(INCOMPATIBLE_TYPEVAR_TO_FUNC.format(type_name) ,severity='fatal')
            elif arg.dtype is not NativeReal():
                args.append(self._print(NumpyFloat(arg)))
            else :
                args.append(self._print(arg))
        code_args = ', '.join(args)
        return '{0}({1})'.format(func_name, code_args)

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
            self._additional_imports.add('pyc_math')
        else:
            if expr.dtype is NativeComplex():
                self._additional_imports.add('cmath')
            else:
                self._additional_imports.add('math')
        args = []
        for arg in expr.args:
            if arg.dtype != expr.dtype:
                cast_func = python_builtin_datatypes[str_dtype(expr.dtype)]
                args.append(self._print(cast_func(arg)))
            else:
                args.append(self._print(arg))
        code_args = ', '.join(args)
        return '{0}({1})'.format(func_name, code_args)

    def _print_MathCeil(self, expr):
        """Convert a Python expression with a math ceil function call to C
        function call"""
        # add necessary include
        self._additional_imports.add('math')
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(PythonFloat(arg))
        else:
            code_arg = self._print(arg)
        return "ceil({})".format(code_arg)

    def _print_MathIsfinite(self, expr):
        """Convert a Python expression with a math isfinite function call to C
        function call"""
        # add necessary include
        self._additional_imports.add('math')
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(PythonFloat(arg))
        else:
            code_arg = self._print(arg)
        return "isfinite({})".format(code_arg)

    def _print_MathIsinf(self, expr):
        """Convert a Python expression with a math isinf function call to C
        function call"""
        # add necessary include
        self._additional_imports.add('math')
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(PythonFloat(arg))
        else:
            code_arg = self._print(arg)
        return "isinf({})".format(code_arg)

    def _print_MathIsnan(self, expr):
        """Convert a Python expression with a math isnan function call to C
        function call"""
        # add necessary include
        self._additional_imports.add('math')
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(PythonFloat(arg))
        else:
            code_arg = self._print(arg)
        return "isnan({})".format(code_arg)

    def _print_MathTrunc(self, expr):
        """Convert a Python expression with a math trunc function call to C
        function call"""
        # add necessary include
        self._additional_imports.add('math')
        arg = expr.args[0]
        if arg.dtype is NativeInteger():
            code_arg = self._print(PythonFloat(arg))
        else:
            code_arg = self._print(arg)
        return "trunc({})".format(code_arg)

    def _print_FunctionAddress(self, expr):
        return expr.name

    def _print_Rand(self, expr):
        raise NotImplementedError("Rand not implemented")

    def _print_NumpyRandint(self, expr):
        raise NotImplementedError("Randint not implemented")

    def _print_Interface(self, expr):
        return ""

    def _print_FunctionDef(self, expr):

        if len(expr.results) > 1:
            self._additional_args.append(expr.results)
        body  = self._print(expr.body)
        decs  = [Declare(i.dtype, i) if isinstance(i, Variable) else FuncAddressDeclare(i) for i in expr.local_vars]
        if len(expr.results) <= 1 :
            decs += [Declare(i.dtype, i) if isinstance(i, Variable) else FuncAddressDeclare(i) for i in expr.results]
        decs += [Declare(i.dtype, i) for i in self._additional_declare]
        decs  = '\n'.join(self._print(i) for i in decs)
        self._additional_declare.clear()

        sep = self._print(SeparatorComment(40))
        if self._additional_args :
            self._additional_args.pop()
        imports = ''.join(self._print(i) for i in expr.imports)
        doc_string = self._print(expr.doc_string) if expr.doc_string else ''

        parts = [sep,
                 doc_string,
                '{signature}\n{{'.format(signature=self.function_signature(expr)),
                 imports,
                 decs,
                 body,
                 '}',
                 sep]

        return '\n'.join(p for p in parts if p)

    def stored_in_c_pointer(self, a):
        if not isinstance(a, Variable):
            return False
        return (a.is_pointer and not a.is_ndarray) or a.is_optional or any(a in b for b in self._additional_args)

    def create_tmp_var(self, match_var):
        tmp_var_name = self._parser.get_new_name('tmp')
        tmp_var = Variable(name = tmp_var_name, dtype = match_var.dtype)
        self._additional_declare.append(tmp_var)
        return tmp_var

    def _print_FunctionCall(self, expr):
        func = expr.funcdef
         # Ensure the correct syntax is used for pointers
        args = []
        for a, f in zip(expr.arguments, func.arguments):
            if isinstance(a, Variable) and self.stored_in_c_pointer(f):
                args.append(VariableAddress(a))
            elif f.is_optional and not isinstance(a, Nil):
                tmp_var = self.create_tmp_var(f)
                assign = Assign(tmp_var, a)
                self._additional_code += self._print(assign) + '\n'
                args.append(VariableAddress(tmp_var))

            else :
                args.append(a)

        args += self._temporary_args
        self._temporary_args = []
        args = ', '.join(['{}'.format(self._print(a)) for a in args])
        if not func.results:
            return '{}({});'.format(func.name, args)
        return '{}({})'.format(func.name, args)

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
        args = [VariableAddress(a) if self.stored_in_c_pointer(a) else a for a in expr.expr]
        if expr.stmt:
            code += self._print(expr.stmt)+'\n'
        if len(args) == 1:
            code +='return {0};'.format(self._print(args[0]))
        elif len(args) > 1:
            code += 'return 0;'
        return code

    def _print_Nil(self, expr):
        return 'NULL'

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
            args = [PythonFloat(a) for a in expr.args]
        else:
            args = expr.args
        return  ' / '.join(self._print(a) for a in args)

    def _print_PyccelFloorDiv(self, expr):
        self._additional_imports.add("math")
        if all(a.dtype is NativeInteger() for a in expr.args):
            args = [PythonFloat(a) for a in expr.args]
        else:
            args = expr.args
        code = ' / '.join(self._print(a) for a in args)
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
        lhs_code = self._print(expr.lhs)
        op = expr.op._symbol
        rhs_code = self._print(expr.rhs)
        return "{0} {1}= {2};".format(lhs_code, op, rhs_code)

    def _print_Assign(self, expr):
        if isinstance(expr.rhs, FunctionCall) and isinstance(expr.rhs.dtype, NativeTuple):
            self._temporary_args = [VariableAddress(a) for a in expr.lhs]
            return '{};'.format(self._print(expr.rhs))
        lhs = self._print(expr.lhs)
        rhs = expr.rhs
        if isinstance(rhs, (NumpyArray)):
            if rhs.rank == 0:
                raise NotImplementedError(expr.lhs + "=" + expr.rhs)
            dummy_array_name, _ = create_incremented_string(self._parser.used_names, prefix = 'array_dummy')
            dtype = self.find_in_dtype_registry(self._print(rhs.dtype), rhs.precision)
            arg = rhs.arg
            if rhs.rank > 1:
                arg = functools.reduce(operator.concat, arg)
            arg = ', '.join(self._print(i) for i in arg)
            dummy_array = "%s %s[] = {%s};\n" % (dtype, dummy_array_name, arg)
            dtype = self.find_in_ndarray_type_registry(format(rhs.dtype), rhs.precision)
            cpy_data = "memcpy({0}.{2}, {1}, {0}.buffer_size);".format(lhs, dummy_array_name, dtype)
            return  '%s%s\n' % (dummy_array, cpy_data)

        if isinstance(rhs, (NumpyFull)):
            code_init = ''
            if rhs.fill_value is not None:
                if isinstance(rhs.fill_value, Literal):
                    dtype = self.find_in_dtype_registry(self._print(rhs.dtype), rhs.precision)
                    code_init = 'array_fill(({0}){1}, {2});'.format(dtype, self._print(rhs.fill_value), lhs)
                else:
                    code_init = 'array_fill({0}, {1});'.format(self._print(rhs.fill_value), lhs)
            else:
                return ''
            return '{}\n'.format(code_init)

        rhs = self._print(rhs)
        return '{} = {};'.format(lhs, rhs)

    def _print_AliasAssign(self, expr):
        lhs = expr.lhs
        rhs = expr.rhs
        if isinstance(rhs, Variable):
            rhs = VariableAddress(rhs)

        lhs = self._print(lhs.name)
        rhs = self._print(rhs)

        # the below condition handles the case of reassinging a pointer to an array view.
        # setting the pointer's is_view attribute to false so it can be ignored by the free_pointer function.
        if isinstance(expr.lhs, Variable) and expr.lhs.is_ndarray \
                and isinstance(expr.rhs, Variable) and expr.rhs.is_ndarray and expr.rhs.is_pointer:
            return 'alias_assign(&{}, {});'.format(lhs, rhs)

        return '{} = {};'.format(lhs, rhs)

    def _print_For(self, expr):
        target = self._print(expr.target)
        body  = self._print(expr.body)
        if isinstance(expr.iterable, PythonRange):
            start, stop, step = [self._print(e) for e in expr.iterable.args]
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        return ('for ({target} = {start}; {target} < {stop}; {target} += '
                '{step})\n{{\n{body}\n}}').format(target=target, start=start,
                stop=stop, step=step, body=body)

    def _print_CodeBlock(self, expr):
        body = []
        for b in expr.body :
            code = self._print(b)
            code = self._additional_code + code
            self._additional_code = ''
            body.append(code)
        return '\n'.join(self._print(b) for b in body)

    def _print_Indexed(self, expr):
        # calculate index for 1d array
        dims = expr.shape
        elem = LiteralInteger(0)
        offset = LiteralInteger(1)
        for i in reversed(list(range(expr.rank))):
            elem += expr.indices[i]*offset
            offset *= dims[i]
        return "%s[%s]" % (self._print(expr.base.label), self._print(elem))

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

    def _handle_is_operator(self, Op, expr):

        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        a = expr.args[0]
        b = expr.args[1]

        if Nil() in expr.args:
            lhs = VariableAddress(expr.lhs) if isinstance(expr.lhs, Variable) else expr.lhs
            rhs = VariableAddress(expr.rhs) if isinstance(expr.rhs, Variable) else expr.rhs

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
                    lines.append("if (%s) {" % self._print(c))
                elif i == len(expr.args) - 1 and c is True:
                    lines.append("else {")
                else:
                    lines.append("else if (%s) {" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                lines.append("}")
            return "\n".join(lines)
        else:
            # The piecewise was used in an expression, need to do inline
            # operators. This has the downside that inline operators will
            # not work for statements that span multiple lines (Matrix or
            # Indexed expressions).
            ecpairs = ["((%s) ? (\n%s\n)\n" % (self._print(c), self._print(e))
                    for e, c in expr.args[:-1]]
            last_line = ": (\n%s\n)" % self._print(expr.args[-1].expr)
            return ": ".join(ecpairs) + last_line + " ".join([")"*len(ecpairs)])

    def _print_MatrixElement(self, expr):
        return "{0}[{1}]".format(expr.parent, expr.j +
                expr.i*expr.parent.shape[1])

    def _print_Variable(self, expr):
        if expr in self._dereference or self.stored_in_c_pointer(expr):
            return '(*{0})'.format(expr.name)
        else:
            return expr.name

    def _print_VariableAddress(self, expr):
        if self.stored_in_c_pointer(expr.variable) or expr.variable.rank > 0:
            return '{}'.format(expr.variable.name)
        else:
            return '&{}'.format(expr.variable.name)

    def _print_Comment(self, expr):
        comments = self._print(expr.text)

        return '/*' + comments + '*/'

    def _print_CommentBlock(self, expr):
        txts = expr.comments
        header = expr.header
        header_size = len(expr.header)

        ln = max(len(i) for i in txts)
        if ln<max(20, header_size+4):
            ln = 20
        top  = '/*' + '_'*int((ln-header_size)/2) + header + '_'*int((ln-header_size)/2) + '*/'
        ln = len(top)-4
        bottom = '/*' + '_'*ln + '*/'

        txts = ['/*' + t + ' '*(ln - len(t)) + '*/' for t in txts]

        body = '\n'.join(i for i in txts)

        return ('{0}\n'
                '{1}\n'
                '{2}').format(top, body, bottom)

    def _print_EmptyNode(self, expr):
        return ''

    def _print_NewLine(self, expr):
        return '\n'

    #=================== OMP ==================
    def _print_OMP_For_Loop(self, expr):
        omp_expr   = str(expr.txt)
        return '#pragma omp for{}\n{{'.format(omp_expr)

    def _print_OMP_Parallel_Construct(self, expr):
        omp_expr   = str(expr.txt)
        return '#pragma omp {}\n{{'.format(omp_expr)

    def _print_OMP_Single_Construct(self, expr):
        omp_expr   = str(expr.txt)
        return '#pragma omp {}\n{{'.format(omp_expr)

    def _print_Omp_End_Clause(self, expr):
        return '}'
    #=====================================

    def _print_Program(self, expr):
        body  = self._print(expr.body)
        decs     = [self._print(i) for i in expr.declarations]
        decs    += [self._print(Declare(i.dtype, i)) for i in self._additional_declare]
        decs    = '\n'.join(self._print(i) for i in decs)
        self._additional_declare.clear()

        # PythonPrint imports last to be sure that all additional_imports have been collected
        imports  = [*expr.imports, *map(Import, self._additional_imports)]
        imports  = '\n'.join(self._print(i) for i in imports)
        return ('{imports}\n'
                'int main()\n{{\n'
                '{decs}\n\n'
                '{body}\n'
                'return 0;\n'
                '}}').format(imports=imports,
                                    decs=decs,
                                    body=body)



    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "    "
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')

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

    _print_Function = CodePrinter._print_not_supported

def ccode(expr, parser, assign_to=None, **settings):
    """Converts an expr to a string of c code

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
    return CCodePrinter(parser, settings).doprint(expr, assign_to)
