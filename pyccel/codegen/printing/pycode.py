# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
# pylint: disable=R0201
# pylint: disable=missing-function-docstring

from itertools import chain

from sympy.printing.pycode import _known_functions
from sympy.printing.pycode import _known_functions_math
from sympy.printing.pycode import _known_constants_math

from pyccel.decorators import __all__ as pyccel_decorators

from pyccel.ast.utilities  import build_types_decorator
from pyccel.ast.core       import CodeBlock, Import, Assign, FunctionCall
from pyccel.ast.datatypes  import default_precision
from pyccel.ast.literals   import LiteralTrue, LiteralString
from pyccel.ast.variable   import DottedName

from pyccel.codegen.printing.codeprinter import CodePrinter

from pyccel.errors.errors import Errors
from pyccel.errors.messages import *

errors = Errors()

#==============================================================================
def _construct_header(func_name, args):
    args = build_types_decorator(args, order='F')
    args = ','.join("{}".format(i) for i in args)
    pattern = '#$ header function static {name}({args})'
    return pattern.format(name=func_name, args=args)

#==============================================================================
class PythonCodePrinter(CodePrinter):
    """A printer to convert pyccel expressions to strings of Python code"""
    printmethod = "_pycode"
    _kf = dict(chain(
        _known_functions.items(),
        [(k, '' + v) for k, v in _known_functions_math.items()]
    ))
    _kc = {k: ''+v for k, v in _known_constants_math.items()}

    def __init__(self, parser=None, **settings):
        self.assert_contiguous = settings.pop('assert_contiguous', False)
        self.parser = parser
        super().__init__(settings=settings)
        self._additional_imports = set()

    def _indent_codestring(self, lines):
        return '    '+lines.replace('\n','\n    ')

    def _format_code(self, lines):
        return lines

    def get_additional_imports(self):
        """return the additional imports collected in printing stage"""
        return self._additional_imports

    def _print_tuple(self, expr):
        fs = ', '.join(self._print(f) for f in expr)
        return '({0})'.format(fs)

    def _print_NativeBool(self, expr):
        return 'bool'

    def _print_NativeInteger(self, expr):
        return 'int'

    def _print_NativeReal(self, expr):
        return 'float'

    def _print_NativeComplex(self, expr):
        return 'complex'

    def _print_Variable(self, expr):
        return self._print(expr.name)

    def _print_ValuedArgument(self, expr):
        return '{} = {}'.format(self._print(expr.argument), self._print(expr.value))

    def _print_Idx(self, expr):
        return self._print(expr.name)

    def _print_IndexedElement(self, expr):
        indices = expr.indices
        if isinstance(indices, (tuple, list)):
            # this a fix since when having a[i,j] the generated code is a[(i,j)]
            if len(indices) == 1 and isinstance(indices[0], (tuple, list)):
                indices = indices[0]

            indices = [self._print(i) for i in indices]
            indices = ','.join(i for i in indices)
        else:
            errors.report(PYCCEL_RESTRICTION_TODO, symbol=expr,
                severity='fatal')

        base = self._print(expr.base)
        return '{base}[{indices}]'.format(base=base, indices=indices)

    def _print_FunctionDef(self, expr):
        name = self._print(expr.name)
        body = self._print(expr.body)
        body = self._indent_codestring(body)
        args = ', '.join(self._print(i) for i in expr.arguments)

        imports = '\n'.join(self._print(i) for i in expr.imports)
        imports = self._indent_codestring(imports)

        doc_string = self._print(expr.doc_string) if expr.doc_string else ''
        doc_string = self._indent_codestring(doc_string)

        code = ('def {name}({args}):\n'
                '{doc_string}\n'
                '\n{imports}\n{body}\n').format(
                        name=name,
                        args=args,
                        doc_string=doc_string,
                        imports=imports,
                        body=body)
        decorators = expr.decorators
        if decorators:
            if decorators['template']:
                # Eliminate template_dict because it is useless in the printing
                expr.decorators['template'] = expr.decorators['template']['decorator_list']
            else:
                expr.decorators.pop('template')
            for n,f in decorators.items():
                if n in pyccel_decorators:
                    self._additional_imports.add(Import(DottedName('pyccel.decorators'), n))
                # TODO - All decorators must be stored in a list
                if not isinstance(f, list):
                    f = [f]
                dec = ''
                for func in f:
                    if isinstance(func, FunctionCall):
                        args = func.args
                    elif func == n:
                        args = []
                    else:
                        args = [LiteralString(a) for a in func]
                    if n == 'types' and len(args)==0:
                        continue
                    if args:
                        args = ', '.join(self._print(i) for i in args)
                        dec += '@{name}({args})\n'.format(name=n, args=args)

                    else:
                        dec += '@{name}\n'.format(name=n)

                code = '{dec}{code}'.format(dec=dec, code=code)
        headers = expr.headers
        if headers:
            headers = self._print(headers)
            code = '{header}\n{code}'.format(header=header, code=code)

        return code

    def _print_FunctionAddress(self, expr):
        return expr.name

    def _print_Return(self, expr):

        rhs_list = [i.rhs for i in expr.stmt.body if isinstance(i, Assign)] if expr.stmt else []
        lhs_list = [i.lhs for i in expr.stmt.body if isinstance(i, Assign)] if expr.stmt else []
        expr_return_vars = [a for a in expr.expr if a not in lhs_list]

        return 'return ' + ','.join(self._print(i) for i in expr_return_vars + rhs_list)

    def _print_Program(self, expr):
        body  = self._print(expr.body)
        body = self._indent_codestring(body)
        imports  = [*expr.imports, *self._additional_imports]
        imports  = '\n'.join(self._print(i) for i in imports)

        return ('{imports}\n'
                'if __name__ == "__main__":\n'
                '{body}\n').format(imports=imports,
                                    body=body)


    def _print_AsName(self, expr):
        name = self._print(expr.name)
        target = self._print(expr.target)

        return '{name} as {target}'.format(name = name, target = target)

    def _print_PythonTuple(self, expr):
        args = ', '.join(self._print(i) for i in expr.args)
        if len(args) == 1:
            args += ','
        return '('+args+')'

    def _print_PythonList(self, expr):
        args = ', '.join(self._print(i) for i in expr.args)
        return '['+args+']'

    def _print_PythonLen(self, expr):
        return 'len({})'.format(self._print(expr.arg))

    def _print_PythonBool(self, expr):
        return 'bool({})'.format(self._print(expr.arg))

    def _print_PythonInt(self, expr):
        type_name = type(expr).__name__.lower()
        is_numpy = type_name.startswith('numpy')
        precision = str(expr.precision*8) if is_numpy else ''
        return 'int{}({})'.format(precision, self._print(expr.arg))

    def _print_PythonFloat(self, expr):
        type_name = type(expr).__name__.lower()
        is_numpy = type_name.startswith('numpy')
        precision = str(expr.precision*8) if is_numpy else ''
        return 'float{}({})'.format(precision, self._print(expr.arg))

    def _print_PythonComplex(self, expr):
        if expr.is_cast:
            return 'complex({})'.format(self._print(expr.internal_var))
        else:
            return 'complex({}, {})'.format(self._print(expr.real), self._print(expr.imag))

    def _print_NumpyComplex(self, expr):
        precision = str(expr.precision*16)
        if expr.is_cast:
            return 'complex{}({})'.format(precision, self._print(expr.internal_var))
        else:
            return 'complex{}({}+{}*1j)'.format(precision, self._print(expr.real), self._print(expr.imag))

    def _print_PythonRange(self, expr):
        return 'range({start}, {stop}, {step})'.format(
                start = self._print(expr.start),
                stop  = self._print(expr.stop ),
                step  = self._print(expr.step ))

    def _print_PythonReal(self, expr):
        return '({}).real'.format(self._print(expr.internal_var))

    def _print_PythonImag(self, expr):
        return '({}).imag'.format(self._print(expr.internal_var))

    def _print_PythonPrint(self, expr):
        return 'print({})'.format(', '.join(self._print(a) for a in expr.expr))

    def _print_PyccelArraySize(self, expr):
        arg = self._print(expr.arg)
        index = self._print(expr.index)
        return 'shape({0})[{1}]'.format(arg, index)

    def _print_Comment(self, expr):
        txt = self._print(expr.text)
        return '# {0} '.format(txt)

    def _print_CommentBlock(self, expr):
        txt = '\n'.join(self._print(c) for c in expr.comments)
        return '"""{0}"""'.format(txt)

    def _print_EmptyNode(self, expr):
        return ''

    def _print_DottedName(self, expr):
        return '.'.join(self._print(n) for n in expr.name)

    def _print_FunctionCall(self, expr):
        if expr.interface:
            func_name = expr.interface_name
        else:
            func_name = expr.funcdef.name
        args = ', '.join(self._print(i) for i in expr.args)
        return'{func}({args})'.format(func=func_name, args=args)

    def _print_Len(self, expr):
        return 'len({})'.format(self._print(expr.arg))

    def _print_Import(self, expr):
        source = self._print(expr.source)
        if not expr.target:
            return 'import {source}'.format(source=source)
        else:
            target = [self._print(i) for i in expr.target]
            if source == "numpy":
                target = [t[:-5] if t.endswith('_like') else t for t in target]
            target = ', '.join(target)
            return 'from {source} import {target}'.format(source=source, target=target)

    def _print_CodeBlock(self, expr):
        if len(expr.body)==0:
            return 'pass'
        else:
            code = '\n'.join(self._print(c) for c in expr.body)
            return code

    def _print_For(self, expr):
        iterable = self._print(expr.iterable)
        target   = expr.target
        if not isinstance(target,(list, tuple)):
            target = [target]
        target = ','.join(self._print(i) for i in target)
        body   = self._print(expr.body)
        body   = self._indent_codestring(body)
        code   = ('for {0} in {1}:\n'
                '{2}\n').format(target,iterable,body)

        return code

    def _print_While(self, expr):
        cond = self._print(expr.test)
        body = self._indent_codestring(self._print(expr.body))
        return 'while {cond}:\n{body}\n'.format(
                cond = cond,
                body = body)

    def _print_Break(self, expr):
        return 'break'

    def _print_Continue(self, expr):
        return 'continue'

    def _print_Assign(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return'{0} = {1}'.format(lhs,rhs)

    def _print_AliasAssign(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return'{0} = {1}'.format(lhs,rhs)

    def _print_AugAssign(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        op  = self._print(expr.op._symbol)
        return'{0} {1}= {2}'.format(lhs,op,rhs)

    def _print_PythonRange(self, expr):
        start = self._print(expr.start)
        stop  = self._print(expr.stop)
        step  = self._print(expr.step)
        return 'range({}, {}, {})'.format(start,stop,step)

    def _print_Product(self, expr):
        args = ','.join(self._print(i) for i in expr.elements)
        return 'product({})'.format(args)

    def _print_Allocate(self, expr):
        return ''

    def _print_Deallocate(self, expr):
        return ''

    def _print_NumpyArray(self, expr):
        dtype = self._print(expr.dtype)
        if expr.precision != default_precision[str(expr.dtype)]:
            factor = 16 if dtype == 'complex' else 8
            dtype += str(expr.precision*factor)

        return "array({arg}, dtype={dtype}, order='{order}')".format(
                arg   = self._print(expr.arg),
                dtype = dtype,
                order = expr.order)

    def _print_NumpyAutoFill(self, expr):
        type_name = type(expr).__name__
        func_name = type_name[5:].lower()

        dtype = self._print(expr.dtype)
        if expr.precision != default_precision[str(expr.dtype)]:
            factor = 16 if dtype == 'complex' else 8
            dtype += str(expr.precision*factor)

        return "{func_name}({shape}, dtype={dtype}, order='{order}')".format(
                func_name = func_name,
                shape = self._print(expr.shape),
                dtype = dtype,
                order = expr.order)

    def _print_NumpyFull(self, expr):
        dtype = self._print(expr.dtype)
        if expr.precision != default_precision[str(expr.dtype)]:
            factor = 16 if dtype == 'complex' else 8
            dtype += str(expr.precision*factor)

        return "full({shape}, {fill_value}, dtype={dtype}, order='{order}')".format(
                shape = self._print(expr.shape),
                fill_value = self._print(expr.fill_value),
                dtype = dtype,
                order = expr.order)

    def _print_NumpyArange(self, expr):
        return "arange({start}, {stop}, {step}, dtype={dtype})".format(
                start = self._print(expr.start),
                stop  = self._print(expr.stop),
                step  = self._print(expr.step),
                dtype = self._print(expr.dtype))

    def _print_NumpySum(self, expr):
        return "sum({})".format(self._print(expr.arg))

    def _print_NumpyUfuncBase(self, expr):
        type_name = type(expr).__name__
        name = type_name[5:].lower()
        args = ', '.join(self._print(a) for a in expr.args)
        return "{}({})".format(name, args)

    def _print_MathFunctionBase(self, expr):
        type_name = type(expr).__name__
        name = type_name[4:].lower()
        args = ', '.join(self._print(a) for a in expr.args)
        return "{}({})".format(name, args)

    def _print_Max(self, expr):
        args = ', '.join(self._print(e) for e in expr.args)
        return 'max({})'.format(args)

    def _print_Min(self, expr):
        args = ', '.join(self._print(e) for e in expr.args)
        return 'min({})'.format(args)

    def _print_Slice(self, expr):
        start = self._print(expr.start) if expr.start else ''
        stop  = self._print(expr.stop)  if expr.stop  else ''
        step  = self._print(expr.step)  if expr.step  else ''
        return '{start}:{stop}:{step}'.format(
                start = start,
                stop  = stop,
                step  = step)

    def _print_Nil(self, expr):
        return 'None'

    def _print_Pass(self, expr):
        return 'pass'

    def _print_PyccelIs(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return'{0} is {1}'.format(lhs,rhs)

    def _print_PyccelIsNot(self, expr):
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)
        return'{0} is not {1}'.format(lhs,rhs)

    def _print_If(self, expr):
        lines = []
        for i, (c, e) in enumerate(expr.blocks):
            if i == 0:
                lines.append("if %s:" % self._print(c))

            elif i == len(expr.blocks) - 1 and isinstance(c, LiteralTrue):
                lines.append("else:")

            else:
                lines.append("elif %s:" % self._print(c))

            if isinstance(e, CodeBlock):
                body = self._indent_codestring(self._print(e))
                lines.append(body)
            else:
                lines.append(self._print(e))
        return "\n".join(lines)

    def _print_IfTernaryOperator(self, expr):
        cond = self._print(expr.cond)
        value_true = self._print(expr.value_true)
        value_false = self._print(expr.value_false)
        return '{true} if {cond} else {false}'.format(cond = cond, true =value_true, false = value_false)

    def _print_Literal(self, expr):
        return repr(expr.python_value)

    def _print_Shape(self, expr):
        arg = self._print(expr.arg)
        if expr.index is None:
            return '{}.shape'.format(arg)

        else:
            index = self._print(expr.index)
            return '{0}.shape[{1}]'.format(arg, index)

    def _print_Print(self, expr):
        args = []
        for f in expr.expr:
            if isinstance(f, str):
                args.append("'{}'".format(f))

            elif isinstance(f, tuple):
                for i in f:
                    args.append(self._print(i))

            else:
                args.append(self._print(f))

        fs = ', '.join(i for i in args)

        return 'print({0})'.format(fs)

    def _print_Module(self, expr):
        # Print interface functions (one function with multiple decorators describes the problem)
        interfaces = [i.functions[0] for i in expr.interfaces]
        for f,i in zip(interfaces, expr.interfaces):
            f.rename(i.name)
        interfaces = '\n'.join(self._print(i) for i in interfaces)
        # Collect functions which are not in an interface
        funcs = [f for f in expr.funcs if not any(f in i.functions for i in expr.interfaces)]
        funcs = '\n'.join(self._print(f) for f in funcs)
        classes = '\n'.join(self._print(c) for c in expr.classes)
        body = '\n'.join((interfaces, funcs, classes))
        imports  = [*expr.imports, *self._additional_imports]
        imports  = '\n'.join(self._print(i) for i in imports)
        return ('{imports}\n\n'
                '{body}').format(
                        imports = imports,
                        body    = body)

    def _print_PyccelPow(self, expr):
        base = self._print(expr.args[0])
        e    = self._print(expr.args[1])
        return '{} ** {}'.format(base, e)

    def _print_PyccelAdd(self, expr):
        return ' + '.join(self._print(a) for a in expr.args)

    def _print_PyccelMinus(self, expr):
        return ' - '.join(self._print(a) for a in expr.args)

    def _print_PyccelMul(self, expr):
        return ' * '.join(self._print(a) for a in expr.args)

    def _print_PyccelDiv(self, expr):
        return ' / '.join(self._print(a) for a in expr.args)

    def _print_PyccelMod(self, expr):
        return '%'.join(self._print(a) for a in expr.args)

    def _print_PyccelFloorDiv(self, expr):
        return '//'.join(self._print(a) for a in expr.args)

    def _print_PyccelAssociativeParenthesis(self, expr):
        return '({})'.format(self._print(expr.args[0]))

    def _print_PyccelUnary(self, expr):
        return '+{}'.format(self._print(expr.args[0]))

    def _print_PyccelUnarySub(self, expr):
        return '-{}'.format(self._print(expr.args[0]))

    def _print_PyccelAnd(self, expr):
        return ' and '.join(self._print(a) for a in expr.args)

    def _print_PyccelOr(self, expr):
        return ' or '.join(self._print(a) for a in expr.args)

    def _print_PyccelEq(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} == {1} '.format(lhs, rhs)

    def _print_PyccelNe(self, expr):
        lhs = self._print(expr.args[0])
        rhs = self._print(expr.args[1])
        return '{0} != {1} '.format(lhs, rhs)

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
        return 'not {}'.format(a)

    def _print_PyccelInvert(self, expr):
        return '~{}'.format(self._print(expr.args[0]))

    def _print_PyccelRShift(self, expr):
        return '{} >> {}'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelLShift(self, expr):
        return '{} << {}'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitXor(self, expr):
        return '{} ^ {}'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitOr(self, expr):
        return '{} | {}'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelBitAnd(self, expr):
        return '{} & {}'.format(self._print(expr.args[0]), self._print(expr.args[1]))

    def _print_PyccelSymbol(self, expr):
        return expr

#==============================================================================
def pycode(expr, assign_to=None, **settings):
    """ Converts an expr to a string of Python code
    Parameters
    ==========
    expr : Expr
        A SymPy expression.
    fully_qualified_modules : bool
        Whether or not to write out full module names of functions
        (``math.sin`` vs. ``sin``). default: ``True``.
    Examples
    ========
    >>> from sympy import tan, Symbol
    >>> from sympy.printing.pycode import pycode
    >>> pycode(tan(Symbol('x')) + 1)
    'math.tan(x) + 1'
    """
    settings.pop('parser', None)
    return PythonCodePrinter(settings).doprint(expr, assign_to)
