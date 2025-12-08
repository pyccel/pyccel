# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Functions for printing C++ code.
"""
from itertools import chain
from pyccel.ast.c_concepts import ObjectAddress
from pyccel.ast.core     import Assign, Declare, Import, Module, AsName
from pyccel.ast.datatypes import PrimitiveIntegerType, PrimitiveBooleanType, PrimitiveFloatingPointType
from pyccel.ast.datatypes import PrimitiveComplexType
from pyccel.ast.datatypes import PythonNativeBool, PythonNativeFloat
from pyccel.ast.datatypes import FinalType
from pyccel.ast.datatypes import HomogeneousSetType, DictType, InhomogeneousTupleType
from pyccel.ast.literals import Nil, LiteralTrue, LiteralString
from pyccel.ast.low_level_tools import UnpackManagedMemory
from pyccel.ast.numpyext import NumpyFloat
from pyccel.ast.utilities import expand_to_loops
from pyccel.ast.variable import Variable, DottedName
from pyccel.codegen.printing.codeprinter import CodePrinter

from pyccel.errors.errors   import Errors
from pyccel.errors.messages import PYCCEL_RESTRICTION_IS_ISNOT, PYCCEL_RESTRICTION_TODO

errors = Errors()

cpp_imports = {n : Import(n, Module(n, (), ())) for n in
                ['cassert',
                 'complex',
                 'cmath',
                 'iostream',
                 'pyc_math_cpp',
                 'cstdint',
                 'string',
                 'tuple']}

# dictionary mapping Math function to (argument_conditions, C_function).
# Used in CCodePrinter._print_MathFunctionBase(self, expr)
# Math function ref https://docs.python.org/3/library/math.html
math_function_to_c = {
    # ---------- Number-theoretic and representation functions ------------
    'MathCeil'     : 'ceil',
    # 'MathComb'   : TODO
    'MathCopysign': 'copysign',
    'MathFabs'   : 'fabs',
    'MathFloor'    : 'floor',
    # 'MathFmod'   : TODO
    # 'MathRexp'   : TODO
    # 'MathFsum'   : TODO
    # 'MathIsclose' : TODO
    'MathIsfinite': 'isfinite',
    'MathIsinf'   : 'isinf',
    'MathIsnan'   : 'isnan',
    # 'MathIsqrt'  : TODO
    'MathLdexp'  : 'ldexp',
    # 'MathModf'  : TODO
    # 'MathPerm'  : TODO
    # 'MathProd'  : TODO
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
    # 'MathDist'  : '???'
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
    # --------------------------- cmath functions --------------------------
    'CmathAcos'  : 'cacos',
    'CmathAcosh' : 'cacosh',
    'CmathAsin'  : 'casin',
    'CmathAsinh' : 'casinh',
    'CmathAtan'  : 'catan',
    'CmathAtanh' : 'catanh',
    'CmathCos'   : 'ccos',
    'CmathCosh'  : 'ccosh',
    'CmathExp'   : 'cexp',
    'CmathSin'   : 'csin',
    'CmathSinh'  : 'csinh',
    'CmathSqrt'  : 'csqrt',
    'CmathTan'   : 'ctan',
    'CmathTanh'  : 'ctanh',
}

cpp_library_headers = {
    "complex",
    "cmath",
    "inttypes",
    "iostream",
    "string",
    "tuple",
}

class CppCodePrinter(CodePrinter):
    """
    A printer for printing code in C.

    A printer to convert Pyccel's AST to strings of c code.
    As for all printers the navigation of this file is done via _print_X
    functions.

    Parameters
    ----------
    filename : str
            The name of the file being pyccelised.
    verbose : int
        The level of verbosity.
    prefix_module : str
            A prefix to be added to the name of the module.
    """
    printmethod = "_cppcode"
    language = "C++"

    _default_settings = {
        'tabwidth': 4,
    }

    def __init__(self, filename, *, verbose):

        errors.set_target(filename)

        super().__init__(verbose)

        self._additional_imports = {}
        self._additional_code = ''
        self._in_header = False
        self._declared_vars = []

    def set_scope(self, scope):
        self._declared_vars.append(set())
        super().set_scope(scope)

    def exit_scope(self):
        super().exit_scope()
        self._declared_vars.pop()

    def _indent_codestring(self, code):
        """
        Indent code to the expected indentation.

        Indent code to the expected indentation.

        Parameters
        ----------
        code : str
            The code to be printed.

        Returns
        -------
            The indented code to be printed.
        """
        tab = ' '*self._default_settings['tabwidth']
        if code == '':
            return code
        else:
            # code ends with \n
            return tab+code.replace('\n','\n'+tab).rstrip(' ')

    def _format_code(self, lines):
        return lines

    def function_signature(self, expr, print_arg_names = True):
        """
        Get the C++ representation of the function signature.

        Extract from the function definition `expr` all the
        information (name, input, output) needed to create the
        function signature and return a string describing the
        function.

        This is not a declaration as the signature does not end
        with a semi-colon.

        Parameters
        ----------
        expr : FunctionDef
            The function definition for which a signature is needed.

        print_arg_names : bool, default : True
            Indicates whether argument names should be printed.

        Returns
        -------
        str
            Signature of the function.
        """
        name = expr.name
        result_var = expr.results.var

        args = ', '.join(self._print(a) for a in expr.arguments)

        result = 'void' if result_var is Nil() else self._print(result_var.class_type)

        return f'{result} {name}({args})'

    def get_declare_type(self, var):
        """
        Get the type of a variable for its declaration.

        Get the type of a variable for its declaration.

        Parameters
        ----------
        var : Variable
            The variable to be declared.

        Returns
        -------
        str
            The code describing the type of the variable.
        """
        class_type = var.class_type
        class_type_str = self._print(class_type)
        const = ' const' if isinstance(class_type, FinalType) else ''

        return f'{class_type_str}{const}'

    def _handle_is_operator(self, Op, expr):
        """
        Get the code to print an `is` or `is not` expression.

        Get the code to print an `is` or `is not` expression. These two operators
        function similarly so this helper function reduces code duplication.

        Parameters
        ----------
        Op : str
            The C operator representing "is" or "is not".

        expr : PyccelIs/PyccelIsNot
            The expression being printed.

        Returns
        -------
        str
            The code describing the expression.

        Raises
        ------
        PyccelError : Raised if the comparison is poorly defined.
        """

        lhs, rhs = expr.args

        if Nil() in expr.args:
            lhs = ObjectAddress(expr.lhs) if isinstance(expr.lhs, Variable) else expr.lhs
            rhs = ObjectAddress(expr.rhs) if isinstance(expr.rhs, Variable) else expr.rhs

            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return f'{lhs_code} {Op} {rhs_code}'

        if (lhs.dtype is PythonNativeBool() and rhs.dtype is PythonNativeBool()):
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return f'{lhs_code} {Op} {rhs_code}'
        else:
            raise errors.report(PYCCEL_RESTRICTION_IS_ISNOT,
                          symbol=expr, severity='fatal')

    def _cast_to(self, expr, dtype):
        """
        Add a cast to an expression when needed.

        Get a format string which provides the code to cast the object `expr`
        to the specified dtype. If the dtypes already
        match then the format string will simply print the expression.

        Parameters
        ----------
        expr : TypedAstNode
            The expression to be cast.
        dtype : PyccelType
            The target type of the cast.

        Returns
        -------
        str
            A format string that contains the desired cast type.
            NB: You should insert the expression to be cast in the string
            after using this function.
        """
        if expr.dtype != dtype:
            return f'({self._print(dtype)})' + '({})'
        return '{}'

    #-----------------------------------------------------------------------
    #                              Print methods
    #-----------------------------------------------------------------------

    def _print_ModuleHeader(self, expr):
        name = expr.module.name
        self.set_scope(expr.module.scope)
        self._in_header = True

        decls = [Declare(v, external=True, module_variable=True) for v in expr.module.variables if not v.is_private]
        global_variables = ''.join(self._print(d) for d in decls)

        classes = '\n'.join(self._print(classDef) for classDef in expr.module.classes)

        funcs = '\n'.join(f"{self.function_signature(f)};" for f in expr.module.funcs if not f.is_inline)

        # Print imports last to be sure that all additional_imports have been collected
        imports = [i for i in chain(expr.module.imports, self._additional_imports.values()) if not i.ignore]
        #imports = self.sort_imports(imports)
        imports = ''.join(self._print(i) for i in imports)

        self.exit_scope()
        self._in_header = False

        sections = ('#pragma once\n',
                    imports,
                    f'namespace {name} {{\n',
                    global_variables,
                    classes,
                    funcs,
                    '}\n')

        return '\n'.join(s for s in sections if s)

    def _print_Module(self, expr):
        self.set_scope(expr.scope)
        name = expr.name

        global_variables = ''.join([self._print(d) for d in expr.declarations])
        body    = ''.join(self._print(i) for i in expr.body)

        # Print imports last to be sure that all additional_imports have been collected
        imports = Import(self.scope.get_python_name(expr.name), Module(expr.name,(),()))
        imports_code = self._print(imports)
        if 'complex' in self._additional_imports:
            imports_code += 'using namespace std::complex_literals;\n'

        self.exit_scope()

        return ''.join((imports_code,
                        f'namespace {name} {{\n\n',
                        global_variables,
                        body,
                        '\n}\n'))

    def _print_Program(self, expr):
        mod = expr.get_direct_user_nodes(lambda x: isinstance(x, Module))[0]
        name = mod.name
        self.set_scope(expr.scope)
        body = self._print(expr.body)
        variables = self.scope.variables.values()
        decs = ''.join(self._print(Declare(v)) for v in variables)

        imports = [i for i in chain(expr.imports, self._additional_imports.values()) if not i.ignore]
        #imports = self.sort_imports(imports)
        imports = ''.join(self._print(i) for i in imports)
        self.exit_scope()
        return ''.join((imports,
                        f'using namespace {name};\n\n',
                        'int main()\n{\n',
                        decs,
                        body,
                        'return 0;\n}'))

    def _print_FunctionDef(self, expr):
        if expr.is_inline:
            return ''

        self.set_scope(expr.scope)

        body  = self._print(expr.body)

        self.exit_scope()

        return ''.join((self.function_signature(expr),
                        ' {\n',
                        self._indent_codestring(body),
                        '}\n'))

    def _print_FunctionDefArgument(self, expr):
        return self.get_declare_type(expr.var) + ' ' + expr.var.name

    def _print_CodeBlock(self, expr):
        if not expr.unravelled:
            body_exprs = expand_to_loops(expr,
                    self.scope.get_temporary_variable, self.scope,
                    language_has_vectors = False)
        else:
            body_exprs = expr.body
        body_code = ''
        for b in body_exprs :
            code = self._print(b)
            code = self._additional_code + code
            self._additional_code = ''
            body_code += code
        return body_code

    def _print_Pass(self, expr):
        return '// pass\n'

    def _print_Return(self, expr):
        if expr.stmt:
            to_print = [l for l in expr.stmt.body if not ((isinstance(l, Assign) and isinstance(l.lhs, Variable))
                                                        or isinstance(l, UnpackManagedMemory))]
            assigns = {a.lhs: a.rhs for a in expr.stmt.body if (isinstance(a, Assign) and isinstance(a.lhs, Variable))}
            assigns.update({a.out_ptr: a.managed_object for a in expr.stmt.body if isinstance(a, UnpackManagedMemory)})
            prelude = ''.join(self._print(l) for l in to_print)
        else:
            assigns = {}
            prelude = ''

        if expr.expr is None:
            return 'return;\n'

        def get_return_code(return_var):
            """ Recursive method which replaces any variables in a return statement whose
            definition is known (via the assigns dict) with the definition. A function is
            required to handle the recursivity implied by an unknown depth of inhomogeneous
            tuples.
            """
            if isinstance(return_var.class_type, InhomogeneousTupleType):
                elem_code = [get_return_code(self.scope.collect_tuple_element(elem)) for elem in return_var]
                return_expr = ', '.join(elem_code)
                if len(elem_code) == 1:
                    return_expr += ','
                return f'std::make_tuple({return_expr})'
            else:
                return_expr = assigns.get(return_var, return_var)
                return self._print(return_expr)

        return prelude + f'return {get_return_code(expr.expr)};\n'

    def _print_Assign(self, expr):
        lhs = expr.lhs

        prefix = ''
        if lhs in self.scope.variables.values() and lhs not in self._declared_vars[-1]:
            prefix = self.get_declare_type(lhs) + ' '
            self._declared_vars[-1].add(lhs)

        lhs_code = self._print(lhs)
        rhs_code = self._print(expr.rhs)
        return f'{prefix}{lhs_code} = {rhs_code};\n'

    # ------------------------------
    #  Ternary operator
    # ------------------------------

    def _print_IfTernaryOperator(self, expr):
        """
        Python: a if cond else b
        C++:    (cond ? a : b)
        """
        c = self._print(expr.cond)
        a = self._print(expr.if_true)
        b = self._print(expr.if_false)
        return f"({c} ? {a} : {b})"


    # ------------------------------
    #  Arithmetic operators
    # ------------------------------

    def _print_PyccelAdd(self, expr):
        target_dtype = expr.dtype
        a, b = expr.args
        a_code = self._cast_to(a, target_dtype).format(self._print(a))
        b_code = self._cast_to(b, target_dtype).format(self._print(b))
        return f"{a_code} + {b_code}"

    def _print_PyccelMinus(self, expr):
        target_dtype = expr.dtype
        a, b = expr.args
        a_code = self._cast_to(a, target_dtype).format(self._print(a))
        b_code = self._cast_to(b, target_dtype).format(self._print(b))
        return f"{a_code} - {b_code}"

    def _print_PyccelMul(self, expr):
        target_dtype = expr.dtype
        a, b = expr.args
        a_code = self._cast_to(a, target_dtype).format(self._print(a))
        b_code = self._cast_to(b, target_dtype).format(self._print(b))
        return f"{a_code} * {b_code}"

    def _print_PyccelDiv(self, expr):
        target_dtype = expr.dtype
        a, b = expr.args
        a_code = self._cast_to(a, target_dtype).format(self._print(a))
        b_code = self._cast_to(b, target_dtype).format(self._print(b))
        return f"{a_code} / {b_code}"

    def _print_PyccelFloorDiv(self, expr):
        # the result type of the floor division is dependent on the arguments
        # type, if all arguments are integers or booleans the result is integer
        # otherwise the result type is float
        need_to_cast = all(a.dtype.primitive_type in (PrimitiveIntegerType(), PrimitiveBooleanType()) for a in expr.args)
        if need_to_cast:
            self.add_import(cpp_imports['pyc_math_cpp'])
            return f'py_floor_div({self._print(expr.args[0])}, {self._print(expr.args[1])})'

        self.add_import(cpp_imports['cmath'])
        code = ' / '.join(self._print(a if a.dtype.primitive_type is PrimitiveFloatingPointType()
                                        else NumpyFloat(a)) for a in expr.args)
        return f"std::floor({code})"

    def _print_PyccelMod(self, expr):
        self.add_import(cpp_imports['pyc_math_cpp'])
        target_dtype = expr.dtype
        n, base = expr.args
        n_code = self._cast_to(n, target_dtype).format(self._print(n))
        base_code = self._cast_to(base, target_dtype).format(self._print(base))
        return f"pyc_modulo({n_code}, {base_code})"

    def _print_PyccelPow(self, expr):
        self.add_import(cpp_imports['cmath'])
        base, exponent = expr.args
        base_code = self._print(base)
        exponent_code = self._print(exponent)

        dtype = expr.dtype

        try:
            exponent_is_pos_int = exponent.dtype.primitive_type is PrimitiveIntegerType() and exponent > 0
        except TypeError:
            exponent_is_pos_int = False

        if base == 2 and exponent_is_pos_int:
            code = f'2 << {exponent_code}'
            current_dtype = exponent.dtype
        else:
            code = f'std::pow({base_code}, {exponent_code})'
            current_dtype = (dtype if dtype.primitive_type not in (PrimitiveIntegerType(), PrimitiveBooleanType())
                             else PythonNativeFloat())

        if current_dtype != dtype:
            return f'({self._print(dtype)})({code})'
        else:
            return code


    # ------------------------------
    #  Unary operators
    # ------------------------------

    def _print_PyccelUnary(self, expr):
        return f"+{self._print(expr.args[0])}"

    def _print_PyccelUnarySub(self, expr):
        return f"-{self._print(expr.args[0])}"

    def _print_PyccelNot(self, expr):
        return f"!({self._print(expr.args[0])})"

    def _print_PyccelInvert(self, expr):
        # Bitwise invert (~)
        return f"~({self._print(expr.args[0])})"


    # ------------------------------
    #  Logical operators
    # ------------------------------

    def _print_PyccelAnd(self, expr):
        return " && ".join(self._print(a) for a in expr.args)

    def _print_PyccelOr(self, expr):
        return " || ".join(self._print(a) for a in expr.args)


    # ------------------------------
    #  Comparison operators
    # ------------------------------

    def _print_PyccelEq(self, expr):
        a, b = expr.args
        return f"{self._print(a)} == {self._print(b)}"

    def _print_PyccelNe(self, expr):
        a, b = expr.args
        return f"{self._print(a)} != {self._print(b)}"

    def _print_PyccelGt(self, expr):
        a, b = expr.args
        return f"{self._print(a)} > {self._print(b)}"

    def _print_PyccelGe(self, expr):
        a, b = expr.args
        return f"{self._print(a)} >= {self._print(b)}"

    def _print_PyccelLt(self, expr):
        a, b = expr.args
        return f"{self._print(a)} < {self._print(b)}"

    def _print_PyccelLe(self, expr):
        a, b = expr.args
        return f"{self._print(a)} <= {self._print(b)}"

    def _print_PyccelIs(self, expr):
        return self._handle_is_operator("==", expr)

    def _print_PyccelIsNot(self, expr):
        return self._handle_is_operator("!=", expr)

    def _print_PyccelIn(self, expr):
        container_type = expr.container.class_type
        element = self._print(expr.element)
        container = self._print(expr.container)
        if isinstance(container_type, (HomogeneousSetType, DictType)):
            # C++ 20
            return f'{container}.contains({element})'
        else:
            # TODO: Lists
            raise errors.report(PYCCEL_RESTRICTION_TODO,
                    symbol = expr,
                    severity='fatal')


    # ------------------------------
    #  Bitwise operators
    # ------------------------------

    def _print_PyccelBitAnd(self, expr):
        return " & ".join(self._print(a) for a in expr.args)

    def _print_PyccelBitOr(self, expr):
        return " | ".join(self._print(a) for a in expr.args)

    def _print_PyccelBitXor(self, expr):
        return " ^ ".join(self._print(a) for a in expr.args)


    # ------------------------------
    #  Bit shifts
    # ------------------------------

    def _print_PyccelLShift(self, expr):
        a, b = expr.args
        return f"{self._print(a)} << {self._print(b)}"

    def _print_PyccelRShift(self, expr):
        a, b = expr.args
        return f"{self._print(a)} >> {self._print(b)}"


    # ------------------------------
    #  Parentheses
    # ------------------------------

    def _print_PyccelAssociativeParenthesis(self, expr):
        return f"({self._print(expr.args[0])})"

    # ------------------------------
    #  Casts
    # ------------------------------

    def _print_PythonFloat(self, expr):
        value = self._print(expr.arg)
        type_name = self._print(expr.dtype)
        return f'({type_name})({value})'

    # ------------------------------
    #  Types
    # ------------------------------

    def _print_PythonNativeBool(self, expr):
        return 'bool'

    def _print_PythonNativeInt(self, expr):
        #TODO: Improve, wrong precision
        return 'int'

    def _print_PythonNativeFloat(self, expr):
        return 'double'

    def _print_PythonNativeComplex(self, expr):
        self.add_import(cpp_imports['complex'])
        return 'std::complex<double>'

    def _print_StringType(self, expr):
        return 'str'

    def _print_NumpyFloat32Type(self, expr):
        return 'float'

    def _print_NumpyFloat64Type(self, expr):
        return 'double'

    def _print_InhomogeneousTupleType(self, expr):
        self.add_import(cpp_imports['tuple'])
        types = ', '.join(self._print(t) for t in expr)
        return f'std::tuple<{types}>'

    # ------------------------------
    #  Mathematical functions
    # ------------------------------

    def _print_MathFunctionBase(self, expr):
        # add necessary include
        type_name = type(expr).__name__
        try:
            func_name = math_function_to_c[type_name]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, severity='fatal')

        if func_name.startswith("pyc"):
            self.add_import(cpp_imports['pyc_math_cpp'])
        else:
            func_name = f'std::{func_name}'
            if expr.dtype.primitive_type is PrimitiveComplexType():
                self.add_import(cpp_imports['complex'])
            else:
                self.add_import(cpp_imports['cmath'])
        if expr.dtype.primitive_type is PrimitiveComplexType():
            args = [self._print(a) for a in expr.args]
        else:
            args = []
            for arg in expr.args:
                if arg.dtype.primitive_type is not PrimitiveFloatingPointType() \
                        and not func_name.startswith("pyc"):
                    args.append(self._print(NumpyFloat(arg)))
                else:
                    args.append(self._print(arg))
        code_args = ', '.join(args)
        if expr.dtype.primitive_type is PrimitiveIntegerType():
            cast_type = self._print(expr.dtype)
            return f'({cast_type}){func_name}({code_args})'
        return f'{func_name}({code_args})'

    # ------------------------------
    #  Literals
    # ------------------------------

    def _print_Literal(self, expr):
        #TODO: Ensure correct precision
        return repr(expr.python_value)

    def _print_LiteralTrue(self, expr):
        return 'true'

    def _print_LiteralFalse(self, expr):
        return 'false'

    def _print_LiteralImaginaryUnit(self, expr):
        self.add_import(cpp_imports['complex'])
        return '1i'

    def _print_LiteralComplex(self, expr):
        if self._in_header:
            return f'{self._print(expr.dtype)}{{{self._print(expr.real)}, {self._print(expr.imag)}}}'
        else:
            if expr.real == 0:
                return self._print(expr.imag) + 'i'
            else:
                return f'({self._print(expr.real)} + {self._print(expr.imag)}i)'

    def _print_LiteralString(self, expr):
        format_str = format(expr.python_value)
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
        return f'"{expr.python_value}"'

    def _print_InhomogeneousTuple(self, expr):
        args = ', '.join(self._print(a) for a in expr)
        return f'std::make_tuple({args})'

    # ------------------------------
    #  Miscellaneous
    # ------------------------------

    def _print_Variable(self, expr):
        name = expr.name
        if expr.is_alias:
            return f'(*{name})'
        else:
            return name

    def _print_Declare(self, expr):
        var = expr.variable

        name = var.name
        class_type = var.class_type
        class_type_str = self._print(class_type)
        const = ' const' if isinstance(class_type, FinalType) else ''

        external = 'extern ' if expr.external else ''
        static = 'static ' if expr.static else ''

        return f'{static}{external}{class_type_str}{const} {name};\n'

    def _print_If(self, expr):
        lines = []
        condition_setup = []
        for i, (c, b) in enumerate(expr.blocks):
            body = self._print(b)
            if i == len(expr.blocks) - 1 and isinstance(c, LiteralTrue):
                if i == 0:
                    lines.append(body)
                    break
                lines.append("else\n")
            else:
                # Print condition
                condition = self._print(c)
                # Retrieve any additional code which cannot be executed in the line containing the condition
                condition_setup.append(self._additional_code)
                self._additional_code = ''
                # Add the condition to the lines of code
                line = f"if ({condition})\n"
                if i == 0:
                    lines.append(line)
                else:
                    lines.append("else " + line)
            lines.append("{\n")
            lines.append(body + "}\n")
        return "".join(chain(condition_setup, lines))

    def _print_Comment(self, expr):
        comments = self._print(expr.text)

        return f'//{comments}\n'

    def _print_Import(self, expr):
        if expr.ignore:
            return ''
        if isinstance(expr.source, AsName):
            source = expr.source.name
        else:
            source = expr.source
        if isinstance(source, DottedName):
            source = source.name[-1].python_value
        else:
            source = self._print(source)

        if source == 'omp_lib':
            source = 'omp'


        if source is None:
            return ''
        if expr.source in cpp_library_headers:
            return f'#include <{source}>\n'
        else:
            return f'#include "{source}.hpp"\n'

    def _print_FunctionCall(self, expr):
        func = expr.funcdef
         # Ensure the correct syntax is used for pointers
        args = [a.value for a in expr.args]

        if func.arguments and func.arguments[0].bound_argument:
            raise NotImplementedError("Classes not yet implemented for C++")

        args = ', '.join(self._print(a) for a in args)

        call_code = f'{func.name}({args})'
        if func.is_imported:
            mod, = func.get_direct_user_nodes(lambda m: isinstance(m, Module))
            call_code = f'{mod.name}::{call_code}'
        if func.results.var is not Nil():
            return call_code
        else:
            return f'{call_code};\n'

    def _print_PythonPrint(self, expr):
        self.add_import(cpp_imports['iostream'])
        end = '\n'
        sep = LiteralString(' ')
        kwargs = [f for f in expr.expr if f.has_keyword]
        for f in kwargs:
            if f.keyword == 'sep'      :   sep = str(f.value)
            elif f.keyword == 'end'    :   end = str(f.value)
            else: errors.report(f"{f.keyword} not implemented as a keyworded argument", severity='fatal')

        args = [f.value for f in expr.expr if not f.has_keyword]
        join_str = ' << {self._print(sep)} << ' if sep != '' else ' << '
        args_str = join_str.join(self._print(a) for a in args)
        if end != '':
            if end == '\n':
                args_str += ' << std::endl;\n'
            else:
                args_str += ' << {self._print(end)};\n'
        else:
            args_str += ';\n'
        return 'std::cout << ' + args_str
