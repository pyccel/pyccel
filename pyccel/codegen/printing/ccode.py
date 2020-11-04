# coding: utf-8
# pylint: disable=R0201
# pylint: disable=missing-function-docstring

from pyccel.ast.numbers   import BooleanTrue, ImaginaryUnit, Float, Integer
from pyccel.ast.core import Nil, PyccelAssociativeParenthesis
from pyccel.ast.core import Assign, datatype, Variable, Import
from pyccel.ast.core import SeparatorComment, VariableAddress
from pyccel.ast.core import DottedName

from pyccel.ast.core import PyccelAdd, PyccelMul, String

from pyccel.ast.datatypes import default_precision
from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeComplex, NativeReal, NativeTuple


from pyccel.ast.numpyext import NumpyFloat
from pyccel.ast.numpyext import NumpyReal, NumpyImag

from pyccel.ast.builtins  import PythonRange, PythonFloat, PythonComplex
from pyccel.ast.core import FuncAddressDeclare, FunctionCall
from pyccel.ast.core import FunctionAddress
from pyccel.ast.core import Declare, ValuedVariable

from pyccel.codegen.printing.codeprinter import CodePrinter

from pyccel.errors.errors import Errors
from pyccel.errors.messages import (PYCCEL_RESTRICTION_TODO, INCOMPATIBLE_TYPEVAR_TO_FUNC,
                                    PYCCEL_RESTRICTION_IS_ISNOT )

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
    # 'MathFactorial': '???', # TODO
    'MathFloor'    : 'floor',
    # 'MathFmod'   : '???',  # TODO
    # 'MathRexp'   : '???'   TODO requires two output
    # 'MathFsum'   : '???',  # TODO
    # 'MathGcd'   : '???',  # TODO
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
    # 'MathSqrt'   : 'sqrt',    # sqrt is printed using _Print_MathSqrt

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

    # -------------------------- Angular conversion ---------------------------

    # 'MathDegrees': '???',  # TODO
    # 'MathRadians': '???', # TODO

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
}

dtype_registry = {('real',8)    : 'double',
                  ('real',4)    : 'float',
                  ('complex',8) : 'double complex',
                  ('complex',4) : 'float complex',
                  ('int',4)     : 'int',
                  ('int',8)     : 'long',
                  ('int',2)     : 'short int',
                  ('int',1)     : 'char',
                  ('bool',4)    : 'bool'}

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

    def __init__(self, parser, settings={}):

        if parser.filename:
            errors.set_target(parser.filename, 'file')

        prefix_module = settings.pop('prefix_module', None)
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        self._dereference = set(settings.get('dereference', []))
        self.prefix_module = prefix_module
        self._additional_imports = set(['stdlib'])
        self._parser = parser
        self._additional_code = ''
        self._additional_declare = []
        self._additional_args = []
        self._temporary_args = []

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

    def _print_Complex(self, expr):
        return self._print(PyccelAssociativeParenthesis(PyccelAdd(expr.real,
                        PyccelMul(expr.imag, ImaginaryUnit()))))

    def _print_PythonComplex(self, expr):
        self._additional_imports.add("complex")
        return self._print(PyccelAssociativeParenthesis(PyccelAdd(expr.real_part,
                        PyccelMul(expr.imag_part, ImaginaryUnit()))))

    def _print_ImaginaryUnit(self, expr):
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
            elif i == len(expr.args) - 1 and c is BooleanTrue():
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

    def _print_BooleanTrue(self, expr):
        return '1'

    def _print_BooleanFalse(self, expr):
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

    def _print_String(self, expr):
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
        args_format = self._print(String(args_format))
        code = ', '.join([args_format, *args])
        return "printf({});".format(code)

    def find_in_dtype_registry(self, dtype, prec):
        try :
            return dtype_registry[(dtype, prec)]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')

    def get_declare_type(self, expr):
        dtype = self._print(expr.dtype)
        prec  = expr.precision
        rank  = expr.rank
        dtype = self.find_in_dtype_registry(dtype, prec)

        if rank > 0:
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
        if expr.variable.rank > 0:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol="rank > 0",severity='fatal')
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
        self._additional_imports.add('math')
        type_name = type(expr).__name__
        try:
            func_name = math_function_to_c[type_name]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')
        args = []
        for arg in expr.args:
            if arg.dtype is NativeComplex():
                self._additional_imports.add('complex')
            if arg.dtype is not NativeReal():
                args.append(self._print(NumpyFloat(arg)))
            else :
                args.append(self._print(arg))
        code_args = ', '.join(args)
        return '{0}({1})'.format(func_name, code_args)

    def _print_MathSqrt(self, expr):
        # add necessary include
        self._additional_imports.add('math')
        arg = expr.args[0]
        if arg.dtype is not NativeReal():
            code_args = self._print(NumpyFloat(arg))
        else :
            code_args = self._print(arg)
        return 'sqrt({})'.format(code_args)

    def _print_FunctionAddress(self, expr):
        return expr.name

    def _print_Rand(self, expr):
        raise NotImplementedError("Rand not implemented")

    def _print_NumpyRandint(self, expr):
        raise NotImplementedError("Randint not implemented")

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

        return ('{sep}\n'
                '{signature}\n{{\n'
                '{imports}\n'
                '{decs}\n'
                '{body}\n'
                '}}\n{sep}'.format(
                    sep = sep,
                    signature = self.function_signature(expr),
                    imports = imports,
                    decs = decs,
                    body = body))

    def stored_in_c_pointer(self, a):
        if not isinstance(a, Variable):
            return False
        return a.is_pointer or a.is_optional or any(a in b for b in self._additional_args)

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
        val = Float(expr.value)
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
        rhs = self._print(expr.rhs)
        return '{} = {};'.format(lhs, rhs)

    def _print_AliasAssign(self, expr):
        lhs = expr.lhs
        rhs = expr.rhs
        if isinstance(rhs, Variable):
            rhs = VariableAddress(rhs)

        lhs = self._print(lhs.name)
        rhs = self._print(rhs)
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
        elem = Integer(0)
        offset = Integer(1)
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

    def _print_NumpyReal(self, expr):
        if expr.arg.dtype is NativeComplex():
            return 'creal({})'.format(self._print(expr.arg))
        else:
            return self._print(expr.arg)

    def _print_NumpyImag(self, expr):
        if expr.arg.dtype is NativeComplex():
            return 'cimag({})'.format(self._print(expr.arg))
        else:
            return '0'

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

    def _print_IsNot(self, expr):
        return self._handle_is_operator("!=", expr)

    def _print_Is(self, expr):
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
        ln = max(len(i) for i in txts)
        if ln<20:
            ln = 20
        top  = '/*' + '_'*int((ln-12)/2) + 'CommentBlock' + '_'*int((ln-12)/2) + '*/'
        ln = len(top)
        bottom = '/*' + '_'*(ln-2) + '*/'

        txts = ['/*' + t + ' '*(ln -2 - len(t)) + '*/' for t in txts]

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
                '{decs}\n'
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
