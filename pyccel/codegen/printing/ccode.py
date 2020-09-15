# coding: utf-8
# pylint: disable=R0201

from sympy.core import S
from sympy.printing.precedence import precedence

from pyccel.ast.numbers   import BooleanTrue, ImaginaryUnit
from pyccel.ast.core import If
from pyccel.ast.core import Assign, datatype, Variable, Import, FunctionCall
from pyccel.ast.core import CommentBlock, Comment, SeparatorComment

from pyccel.ast.core import PyccelPow, PyccelAdd, PyccelMul, PyccelDiv, PyccelMod, PyccelFloorDiv
from pyccel.ast.core import PyccelEq,  PyccelNe,  PyccelLt,  PyccelLe,  PyccelGt,  PyccelGe
from pyccel.ast.core import PyccelAnd, PyccelOr,  PyccelNot, PyccelMinus

from pyccel.ast.datatypes import NativeInteger, NativeBool, NativeComplex

from pyccel.ast.builtins  import Range
from pyccel.ast.core import Declare
from pyccel.ast.core import SeparatorComment

from pyccel.codegen.printing.codeprinter import CodePrinter

from pyccel.errors.errors import Errors
from pyccel.errors.messages import *

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

dtype_registry = {('real',8)    : 'double',
                  ('real',4)    : 'float',
                  ('complex',8) : 'double _Complex',
                  ('complex',4) : 'float _Complex',
                  ('int',4)     : 'int',
                  ('int',8)     : 'long',
                  ('int',2)     : 'short int',
                  ('int',1)     : 'char',
                  ('bool',4)    : '_Bool'}


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

        prefix_module = settings.pop('prefix_module', None)
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        self._dereference = set(settings.get('dereference', []))
        self.prefix_module = prefix_module
        self._additional_imports = set(['stdlib.h'])

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
        return '(double)({0})'.format(value)

    def _print_Int(self, expr):
        value = self._print(expr.arg)
        return '(long)({0})'.format(value)

    def _print_Bool(self, expr):
        value = self._print(expr.arg)
        return '{} != 0'.format(value)

    def _print_Complex(self, expr):
        self._additional_imports.add('complex.h')
        return self._print(PyccelAdd(expr.real,
                        PyccelMul(expr.imag, ImaginaryUnit())))

    def _print_PythonComplex(self, expr):
        self._additional_imports.add('complex.h')
        return self._print(PyccelAdd(expr.real_part,
                        PyccelMul(expr.imag_part, ImaginaryUnit())))

    def _print_ImaginaryUnit(self, expr):
        return '_Complex_I'
    def _print_Module(self, expr):
        body    = '\n\n'.join(self._print(i) for i in expr.body)

        # Print imports last to be sure that all additional_imports have been collected
        imports  = list(expr.imports)
        imports += [Import(s) for s in self._additional_imports]
        imports = ''.join(self._print(i) for i in imports)
        return ('{imports}\n\n'
                '{body}').format(
                        imports = imports,
                        body    = body)

    def _print_While(self,expr):
        code = "while (%s)\n{" % self._print(expr.test)
        code = code + "\n %s" % self._print(expr.body) + "\n}"
        return (code)

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

    def _print_Import(self, expr):
        return '#include "{0}"'.format(expr.source)

    def find_in_dtype_registry(self, dtype, prec):
        try :
            return dtype_registry[(dtype, prec)]
        except KeyError:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,severity='fatal')

    def get_declare_type(self, expr):
        dtype = self._print(expr.dtype)
        prec  = expr.precision
        rank  = expr.rank
        dtype = self.find_in_dtype_registry(dtype,prec)

        if rank > 0 or expr.is_pointer:
            return '{0} *'.format(dtype)
        else:
            return '{0} '.format(dtype)

    def _print_Declare(self, expr):
        declaration_type = self.get_declare_type(expr.variable)
        variable = self._print(expr.variable.name)

        return '{0}{1};'.format(declaration_type, variable)

    def _print_NativeBool(self, expr):
        return 'bool'

    def _print_NativeInteger(self, expr):
        return 'int'

    def _print_NativeReal(self, expr):
        return 'real'

    def _print_NativeVoid(self, expr):
        return 'void'

    def _print_NativeComplex(self, expr):
        return 'complex'

    def function_signature(self, expr):
        if len(expr.results) == 1:
            ret_type = self.get_declare_type(expr.results[0])
        elif len(expr.results) > 1:
            # TODO: Use fortran example to add pointer arguments for multiple output
            msg = 'Multiple output arguments is not yet supported in c'
            errors.report(msg+'\n'+PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal', blocker=self.blocking)
        else:
            ret_type = self._print(datatype('void')) + ' '
        name = expr.name
        if not expr.arguments:
            arg_code = 'void'
        else:
            arg_code = ', '.join('{0}{1}'.format(self.get_declare_type(i), i) for i in expr.arguments)
        return '{0}{1}({2})'.format(ret_type, name, arg_code)

    def _print_FunctionDef(self, expr):

        decs  = [Declare(i.dtype, i) for i in expr.local_vars]
        decs += [Declare(i.dtype, i) for i in expr.results]
        decs  = '\n'.join(self._print(i) for i in decs)
        body  = '\n'.join(self._print(i) for i in expr.body.body)
        sep = self._print(SeparatorComment(40))

        return ('{sep}\n'
                '{signature}\n{{\n'
                '{decs}\n'
                '{body}\n'
                '}}\n{sep}'.format(
                    sep = sep,
                    signature = self.function_signature(expr),
                    decs = decs,
                    body = body))

    def _print_FunctionCall(self, expr):
        func = expr.funcdef
        # currently support only function with one or zero output
        args = ','.join(['{}'.format(self._print(a)) for a in expr.arguments])
        if not func.results:
            return '{}({});'.format(func.name, args)
        return '{}({})'.format(func.name, args)

    def _print_Return(self, expr):
        code = ''
        if expr.stmt:
            code += self._print(expr.stmt)+'\n'
        code +='return {0};'.format(self._print(expr.expr[0]))
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
        args = [self._print(a) for a in expr.args]
        if all(a.dtype is NativeInteger() for a in expr.args):
            return ' / '.join('real({})'.format(self._print(a)) for a in args)
        return  ' / '.join(self._print(a) for a in args)

    def _print_PyccelAssociativeParenthesis(self, expr):
        return '({})'.format(self._print(expr.args[0]))

    def _print_PyccelUnary(self, expr):
        return '({})'.format(self._print(expr.args[0]))

    def _print_AugAssign(self, expr):
        lhs_code = self._print(expr.lhs)
        op = expr.op._symbol
        rhs_code = self._print(expr.rhs)
        return "{0} {1}= {2};".format(lhs_code, op, rhs_code)

    def _print_Assign(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return '{} = {};'.format(lhs, rhs)

    def _print_AliasAssign(self, expr):
        lhs = self._print(expr.lhs.name)
        rhs = self._print(expr.rhs)

        return '{} = {};'.format(lhs, rhs)

    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = [self._print(e) for e in expr.iterable.args]
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        body = '\n'.join(self._print(i) for i in expr.body.body)
        return ('for ({target} = {start}; {target} < {stop}; {target} += '
                '{step})\n{{\n{body}\n}}').format(target=target, start=start,
                stop=stop, step=step, body=body)

    def _print_CodeBlock(self, expr):
        return '\n'.join(self._print(b) for b in expr.body)

    def _print_Pow(self, expr):
        if "Pow" in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        if expr.exp == -1:
            return '1.0/%s' % (self.parenthesize(expr.base, PREC))
        elif expr.exp == 0.5:
            return 'sqrt(%s)' % self._print(expr.base)
        else:
            return 'pow(%s, %s)' % (self._print(expr.base),
                                 self._print(expr.exp))

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return '%d.0L/%d.0L' % (p, q)

    def _print_Indexed(self, expr):
        # calculate index for 1d array
        dims = expr.shape
        elem = S.Zero
        offset = S.One
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

    def _print_Bool(self, expr):
        value = self._print(expr.arg)
        return '{} != 0'.format(value)

    def _print_Real(self, expr):
        if expr.arg.dtype is NativeComplex():
            self._additional_imports.add('complex.h')
            return 'creal({})'.format(self._print(expr.arg))
        else:
            return self._print(expr.arg)

    def _print_Imag(self, expr):
        if expr.arg.dtype is NativeComplex():
            self._additional_imports.add('complex.h')
            return 'cimag({})'.format(self._print(expr.arg))
        else:
            return '0'

    def _print_Piecewise(self, expr):
        if expr.args[-1].cond != True:
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
                elif i == len(expr.args) - 1 and c == True:
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
        if expr in self._dereference or expr.is_pointer:
            return '(*{0})'.format(expr.name)
        else:
            return expr.name

    def _print_VariableAddress(self, expr):
        if expr.variable.is_pointer or expr.variable.rank > 0:
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

        for i in range(len(txts)):
            txts[i] = '/*' + txts[i] + ' '*(ln -2 - len(txts[i])) + '*/'


        body = '\n'.join(i for i in txts)

        return ('{0}\n'
                '{1}\n'
                '{2}').format(top, body, bottom)

    def _print_EmptyNode(self, expr):
        return ''

    def _print_NewLine(self, expr):
        return '\n'


    def _print_Program(self, expr):
        body     = '\n'.join(self._print(i) for i in expr.body.body)
        decs     = '\n'.join(self._print(i) for i in expr.declarations)

        # Print imports last to be sure that all additional_imports have been collected
        imports  = list(expr.imports)
        imports += [Import(s) for s in self._additional_imports]
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
