# coding: utf-8

"""Print to F90 standard. Trying to follow the information provided at
www.fortran90.org as much as possible."""

from __future__ import print_function, division
import string
from itertools import groupby

from sympy.core import Symbol
from sympy.core import S, Add, N
from sympy.core import Tuple
from sympy.core.function import Function
from sympy.core.compatibility import string_types
from sympy.printing.precedence import precedence
from sympy.sets.fancysets import Range

from pyccel.types.ast import (Assign, MultiAssign, Result, InArgument,
        OutArgument, InOutArgument, Variable, Declare,LEN)
from pyccel.printers.codeprinter import CodePrinter

__all__ = ["FCodePrinter", "fcode"]

known_functions = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "log": "log",
    "exp": "exp",
    "erf": "erf",
    "Abs": "Abs",
    "sign": "sign",
    "conjugate": "conjg"
}

class FCodePrinter(CodePrinter):
    """A printer to convert sympy expressions to strings of Fortran code"""
    printmethod = "_fcode"
    language = "Fortran"

    _default_settings = {
        'order': None,
        'full_prec': 'auto',
        'precision': 15,
        'user_functions': {},
        'human': True,
        'source_format': 'fixed',
        'contract': True,
        'standard': 77
    }

    _operators = {
        'and': '.and.',
        'or': '.or.',
        'xor': '.neqv.',
        'equivalent': '.eqv.',
        'not': '.not. ',
    }

    _relationals = {
        '!=': '/=',
    }

    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)

    def _get_statement(self, codestring):
        return codestring

    def _get_comment(self, text):
        return "! {0}".format(text)

    def _format_code(self, lines):
        return self._wrap_fortran(self.indent_code(lines))

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for j in range(cols) for i in range(rows))

    # ============ Elements ============ #

    def _print_Module(self, expr):
        return '\n\n'.join(self._print(i) for i in expr.body)

    def _print_Import(self, expr):
        fil = self._print(expr.fil)
        if not expr.funcs:
            return 'use {0}'.format(fil)
        else:
            funcs = ', '.join(self._print(f) for f in expr.funcs)
            return 'use {0}, only: {1}'.format(fil, funcs)

    def _print_Print(self, expr):
        Str=[]
        for f in expr.expr:
             if isinstance(f,str):
                 Str.append(repr(f))
             else:
                Str.append(self._print(f))


        fs = ', '.join(Str)

        return 'print *, {0} '.format(fs)

    def _print_Comment(self, expr):
        txt = self._print(expr.text)
        return '! {0} '.format(txt)

    def _print_AnnotatedComment(self, expr):
        accel      = self._print(expr.accel)
        do         = self._print(expr.do)
        end        = self._print(expr.end)
        parallel   = self._print(expr.parallel)
        section    = self._print(expr.section)
        visibility = self._print(expr.visibility)
        variables  = ', '.join(self._print(f) for f in expr.variables)
        if len(variables) > 0:
            variables  = '(' + variables + ')'

        if not do:
            do = ''

        if not end:
            end = ''

        if not parallel:
            parallel = ''

        if not section:
            section = ''

        if not visibility:
            visibility = ''

        return '!${0} {1} {2} {3} {4} {5} {6}'.format(accel, do, end, parallel, section, visibility, variables)

    def _print_Tuple(self, expr):
        fs = ', '.join(self._print(f) for f in expr)
        return '(/ {0} /)'.format(fs)

    def _print_NumpyZeros(self, expr):
        lhs_code   = self._print(expr.lhs)

        if isinstance(expr.shape, Tuple):
#            shape_code = ', '.join(self._print(i) for i in expr.shape)
            shape_code = ', '.join('0:' + self._print(i) + '-1' for i in expr.shape)
        else:
            shape_code = '0:' + self._print(expr.shape) + '-1'

#        return self._get_statement("%s = zeros(%s)" % (lhs_code, shape_code))
        return self._get_statement("allocate(%s(%s)) ; %s = 0" % (lhs_code, shape_code, lhs_code))
    def _print_NumpyOnes(self, expr):
        lhs_code   = self._print(expr.lhs)

        if isinstance(expr.shape, Tuple):
#            shape_code = ', '.join(self._print(i) for i in expr.shape)
            shape_code = ', '.join('0:' + self._print(i) + '-1' for i in expr.shape)
        else:
            shape_code = '0:' + self._print(expr.shape) + '-1'

#        return self._get_statement("%s = zeros(%s)" % (lhs_code, shape_code))
        return self._get_statement("allocate(%s(%s)) ; %s = 1" % (lhs_code, shape_code, lhs_code))


    def _print_NumpyLinspace(self, expr):
        lhs_code   = self._print(expr.lhs)
        start_code = self._print(expr.start)
        end_code   = self._print(expr.end)
        size_code  = self._print(expr.size)

        return self._get_statement("%s = linspace(%s, %s, %s)" % \
                                   (lhs_code, start_code, end_code, size_code))
    def _print_NumpyArray(self,expr):
        lhs_code   = self._print(expr.lhs)
        lhs_size   =self._print(len(expr.rhs))

        return self._get_statement("allocate(%s(%s)) ; %s =( /"%(lhs_code,lhs_size,lhs_code)+','.join(str(i) for i in expr.rhs)+"/ )")

    def _print_LEN(self,expr):
        if isinstance(expr.rhs,list):
            st=','.join([str(i) for i in expr.rhs])
            return self._get_statement('%s =size((/%s/),1)'%(expr.lhs,st))
        else:
            return self._get_statement('%s =size(%s,1)'%(expr.lhs,expr.rhs))
            
    def _print_Declare(self, expr):
        dtype = self._print(expr.dtype)
        intent_lookup = {InArgument: 'in',
                         OutArgument: 'out',
                         InOutArgument: 'inout',
                         Variable: None}
        # Group the variables by intent
        f = lambda x: intent_lookup[type(x)]

        arg_types        = [type(v) for v in expr.variables]
        var_list         = groupby(sorted(expr.variables, key=f), f)
        arg_ranks        = [v.rank for v in expr.variables]
        arg_allocatables = [v.allocatable for v in expr.variables]

        decs = []
        for intent, g in var_list:
            vstr = ', '.join(self._print(i.name) for i in g)

            # TODO ARA improve
            rank        = arg_ranks[0]
            allocatable = arg_allocatables[0]

            if rank == 0:
                rankstr =  ''
            else:
                rankstr = ', '.join(':' for f in range(0, rank))
                rankstr = '(' + rankstr + ')'

            if allocatable:
                allocatablestr = ', allocatable'
            else:
                allocatablestr = ''

            if intent:
                decs.append('{0}, intent({1}) {2} :: {3} {4}'.
                            format(dtype, intent, allocatablestr, vstr, rankstr))
            else:
                decs.append('{0}{1} :: {2} {3}'.
                            format(dtype, allocatablestr, vstr, rankstr))

        return '\n'.join(decs)
    
   
        

    def _print_NativeBool(self, expr):
        return 'logical'

    def _print_NativeInteger(self, expr):
        return 'integer'

    def _print_NativeFloat(self, expr):
        return 'real'

    def _print_NativeDouble(self, expr):
        return 'real(kind=8)'

    def _print_NativeComplex(self, expr):
        # TODO add precision
        return 'complex(kind=8)'

    def _print_EqualityStmt(self, expr):
        return '{0} == {1} '.format(expr.lhs, expr.rhs)

    def _print_NotequalStmt(self, expr):
        return '{0} /= {1} '.format(expr.lhs, expr.rhs)

    def _print_FunctionDef(self, expr):
        name = str(expr.name)
        out_args = []
        decs = []
        body = expr.body
        func_end  = ''
        if len(expr.results) == 1:
            result = expr.results[0]

            body = []
            for stmt in expr.body:
                if isinstance(stmt, Declare):
                    # TODO improve
                    if not(str(stmt.variables[0].name) == str(result.name)):
                        decs.append(stmt)
                elif not isinstance(stmt, list): # for list of Results
                    body.append(stmt)

            ret_type = self._print(result.dtype)
            func_type = 'function'

            if result.allocatable:
                sig = 'function {0}'.format(name)
                for n in [result.name, name]:
                    var = Variable(result.dtype, n, \
                                 rank=result.rank, \
                                 allocatable=result.allocatable, \
                                 shape=result.shape)

                    dec = Declare(result.dtype, var)
                    decs.append(dec)
                body.append(Assign(Symbol(name), result.name))
            else:
                sig = '{0} function {1}'.format(ret_type, name)
                func_end  = ' result({0})'.format(result.name)
        elif len(expr.results) > 1:
            for result in expr.results:
                arg = OutArgument(result.dtype, result.name)
                out_args.append(arg)

                dec = Declare(result.dtype, arg)
                decs.append(dec)
            sig = 'subroutine ' + name
            func_type = 'subroutine'

            names = [str(res.name) for res in expr.results]
            body = []
            for stmt in expr.body:
                if isinstance(stmt, Declare):
                    # TODO improve
                    nm = str(stmt.variables[0].name)
                    if not(nm in names):
                        decs.append(stmt)
                elif not isinstance(stmt, list): # for list of Results
                    body.append(stmt)
        else:
            sig = 'subroutine ' + name
            func_type = 'subroutine'
        out_code  = ', '.join(self._print(i) for i in out_args)

        arg_code  = ', '.join(self._print(i) for i in expr.arguments)
        if len(out_code) > 0:
            arg_code  = ', '.join(i for i in [arg_code, out_code])

        body_code = '\n'.join(self._print(i) for i in body)
        prelude   = '\n'.join(self._print(i) for i in decs)

        body_code = prelude + '\n\n' + body_code

        return ('{0}({1}) {2}\n'
                'implicit none\n'
#                'integer, parameter:: dp=kind(0.d0)\n'
                '{3}\n\n'
                'end {4}').format(sig, arg_code, func_end, body_code, func_type)

    def _print_InArgument(self, expr):
        return self._print(expr.name)

    def _print_OutArgument(self, expr):
        return self._print(expr.name)

    def _print_InOutArgument(self, expr):
        return self._print(expr.name)

    def _print_Return(self, expr):
        return 'return'

    def _print_AugAssign(self, expr):
        raise NotImplementedError("Fortran doesn't support AugAssign")

    def _print_MultiAssign(self, expr):
        # TODO improve, case where no input args, etc ...
        if isinstance(expr.rhs, str):
            args    = ', '.join(self._print(i) for i in expr.trailer)
            outputs = ', '.join(self._print(i) for i in expr.lhs)

            return 'call {0} ({1}, {2})'.format(expr.rhs, args, outputs)
        else:
            raise TypeError("Expecting a string for the rhs.")


    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        # decrement stop by 1 because of the python convention
        stop = stop - 1

        body = '\n'.join(self._print(i) for i in expr.body)
        return ('do {target} = {start}, {stop}, {step}\n'
                '{body}\n'
                'end do').format(target=target, start=start, stop=stop,
                        step=step, body=body)

    def _print_While(self,expr):
        body = '\n'.join(self._print(i) for i in expr.body)
        return ('do while ({test}) \n'
                '{body}\n'
                'end do').format(test=expr.test,body=body)

    def _print_Piecewise(self, expr):
        lines = []
        for i, (c, e) in enumerate(expr.args):
            if i == 0:
                lines.append("if (%s) then" % self._print(c))
            elif i == len(expr.args) - 1 and c == True:
                lines.append("else")
            else:
                lines.append("else if (%s) then" % self._print(c))
            if isinstance(e, list):
                for ee in e:
                    lines.append(self._print(ee))
            else:
                lines.append(self._print(e))
        lines.append("end if")
        return "\n".join(lines)

    def _print_MatrixElement(self, expr):
        return "{0}({1}, {2})".format(expr.parent, expr.i + 1, expr.j + 1)

    def _print_Add(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        # collect the purely real and purely imaginary parts:
        pure_real = []
        pure_imaginary = []
        mixed = []
        for arg in expr.args:
            if arg.is_number and arg.is_real:
                pure_real.append(arg)
            elif arg.is_number and arg.is_imaginary:
                pure_imaginary.append(arg)
            else:
                mixed.append(arg)
        if len(pure_imaginary) > 0:
            if len(mixed) > 0:
                PREC = precedence(expr)
                term = Add(*mixed)
                t = self._print(term)
                if t.startswith('-'):
                    sign = "-"
                    t = t[1:]
                else:
                    sign = "+"
                if precedence(term) < PREC:
                    t = "(%s)" % t

                return "cmplx(%s,%s) %s %s" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                    sign, t,
                )
            else:
                return "cmplx(%s,%s)" % (
                    self._print(Add(*pure_real)),
                    self._print(-S.ImaginaryUnit*Add(*pure_imaginary)),
                )
        else:
            return CodePrinter._print_Add(self, expr)

    def _print_Function(self, expr):
        # All constant function args are evaluated as floats
        prec =  self._settings['precision']
        args = [N(a, prec) for a in expr.args]
        eval_expr = expr.func(*args)
        if not isinstance(eval_expr, Function):
            return self._print(eval_expr)
        else:
            return CodePrinter._print_Function(self, expr.func(*args))

    def _print_ImaginaryUnit(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        return "cmplx(0,1)"

    def _print_int(self, expr):
        return str(expr)

    def _print_Mul(self, expr):
        # purpose: print complex numbers nicely in Fortran.
        if expr.is_number and expr.is_imaginary:
            return "cmplx(0,%s)" % (
                self._print(-S.ImaginaryUnit*expr)
            )
        else:
            return CodePrinter._print_Mul(self, expr)

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if expr.exp == -1:
            return '1.0/%s' % (self.parenthesize(expr.base, PREC))
        elif expr.exp == 0.5:
            if expr.base.is_integer:
                # Fortan intrinsic sqrt() does not accept integer argument
                if expr.base.is_Number:
                    return 'sqrt(%s.0d0)' % self._print(expr.base)
                else:
                    return 'sqrt(dble(%s))' % self._print(expr.base)
            else:
                return 'sqrt(%s)' % self._print(expr.base)
        else:
            return CodePrinter._print_Pow(self, expr)

    def _print_Rational(self, expr):
        p = expr.numerator
        q = expr.denominator
        if type(p) == int:
            txt_p = "%d.0d0" % (p)
        else:
            txt_p = '{} * 1.0d0'.format(str(p))
        if type(q) == int:
            txt_q = "%d.0d0" % (q)
        else:
            txt_q = '{} * 1.0d0'.format(str(q))
        return '{0}/{1}'.format(txt_p, txt_q)

    def _print_Float(self, expr):
        printed = CodePrinter._print_Float(self, expr)
        e = printed.find('e')
        if e > -1:
            return "%sd%s" % (printed[:e], printed[e + 1:])
        return "%sd0" % printed

    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s(%s)" % (self._print(expr.base.label), ", ".join(inds))

    def _print_Idx(self, expr):
        return self._print(expr.label)

    def _print_Slice(self, expr):
        return "%s:%s" % (self._print(expr.start), self._print(expr.end))

    def _pad_leading_columns(self, lines):
        result = []
        for line in lines:
            if line.startswith('!'):
                result.append("! " + line[1:].lstrip())
            else:
                result.append(line)
        return result

    def _wrap_fortran(self, lines):
        """Wrap long Fortran lines

           Argument:
             lines  --  a list of lines (without \\n character)

           A comment line is split at white space. Code lines are split with a more
           complex rule to give nice results.
        """
        # routine to find split point in a code line
        my_alnum = set("_+-." + string.digits + string.ascii_letters)
        my_white = set(" \t()")

        def split_pos_code(line, endpos):
            if len(line) <= endpos:
                return len(line)
            pos = endpos
            split = lambda pos: \
                (line[pos] in my_alnum and line[pos - 1] not in my_alnum) or \
                (line[pos] not in my_alnum and line[pos - 1] in my_alnum) or \
                (line[pos] in my_white and line[pos - 1] not in my_white) or \
                (line[pos] not in my_white and line[pos - 1] in my_white)
            while not split(pos):
                pos -= 1
                if pos == 0:
                    return endpos
            return pos
        # split line by line and add the splitted lines to result
        result = []
        trailing = ' &'
        for line in lines:
            if line.startswith("! "):
                # comment line
                if len(line) > 72:
                    pos = line.rfind(" ", 6, 72)
                    if pos == -1:
                        pos = 72
                    hunk = line[:pos]
                    line = line[pos:].lstrip()
                    result.append(hunk)
                    while len(line) > 0:
                        pos = line.rfind(" ", 0, 66)
                        if pos == -1 or len(line) < 66:
                            pos = 66
                        hunk = line[:pos]
                        line = line[pos:].lstrip()
                        result.append("%s%s" % ("! ", hunk))
                else:
                    result.append(line)
            else:
                # code line
                pos = split_pos_code(line, 72)
                hunk = line[:pos].rstrip()
                line = line[pos:].lstrip()
                if line:
                    hunk += trailing
                result.append(hunk)
                while len(line) > 0:
                    pos = split_pos_code(line, 65)
                    hunk = line[:pos].rstrip()
                    line = line[pos:].lstrip()
                    if line:
                        hunk += trailing
                    result.append("%s%s" % ("      " , hunk))
        return result

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""
        if isinstance(code, string_types):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        code = [ line.lstrip(' \t') for line in code ]

        inc_keyword = ('do ', 'if(', 'if ', 'do\n', 'else')
        dec_keyword = ('end do', 'enddo', 'end if', 'endif', 'else')

        increase = [ int(any(map(line.startswith, inc_keyword)))
                     for line in code ]
        decrease = [ int(any(map(line.startswith, dec_keyword)))
                     for line in code ]
        continuation = [ int(any(map(line.endswith, ['&', '&\n'])))
                         for line in code ]

        level = 0
        cont_padding = 0
        tabwidth = 4
        new_code = []
        for i, line in enumerate(code):
            if line == '' or line == '\n':
                new_code.append(line)
                continue
            level -= decrease[i]

            padding = " "*(level*tabwidth + cont_padding)

            line = "%s%s" % (padding, line)

            new_code.append(line)

            if continuation[i]:
                cont_padding = 2*tabwidth
            else:
                cont_padding = 0
            level += increase[i]

        return new_code


def fcode(expr, assign_to=None, **settings):
    """Converts an expr to a string of c code

    Parameters
    ==========

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

    Examples
    ========

    >>> from sympy import fcode, symbols, Rational, sin, ceiling, floor
    >>> x, tau = symbols("x, tau")
    >>> fcode((2*tau)**Rational(7, 2))
    '8*sqrt(2.0d0)*tau**(7.0d0/2.0d0)'
    >>> fcode(sin(x), assign_to="s")
    's = sin(x)'

    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg. Alternatively, the
    dictionary value can be a list of tuples i.e. [(argument_test,
    cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "floor": [(lambda x: not x.is_integer, "FLOOR1"),
    ...             (lambda x: x.is_integer, "FLOOR2")]
    ... }
    >>> fcode(floor(x) + ceiling(x), user_functions=custom_functions)
    'CEIL(x) + FLOOR1(x)'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(fcode(expr, tau))
    if (x > 0) then
        tau = x + 1
    else
        tau = x
    end if

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(fcode(mat, A))
    A(1, 1) = x**2
        if (x > 0) then
    A(2, 1) = x + 1
        else
    A(2, 1) = x
        end if
    A(3, 1) = sin(x)
    """

    return FCodePrinter(settings).doprint(expr, assign_to)
