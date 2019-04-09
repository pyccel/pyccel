from sympy.core import Symbol
from sympy import Tuple

from pyccel.codegen.printing.pycode import PythonCodePrinter as PyccelPythonCodePrinter

class PythonCodePrinter(PyccelPythonCodePrinter):

    def __init__(self, settings=None):
        PyccelPythonCodePrinter.__init__(self, settings=settings)

    def _print_AppliedUndef(self, expr):
        args = ','.join(self._print(i) for i in expr.args)
        fname = self._print(expr.func.__name__)
        return '{fname}({args})'.format(fname=fname, args=args)

    def _print_SequentialBlock(self, expr):
        code = ''

        # ...
        if expr.decs:
            decs = '\n'.join(self._print(i) for i in expr.decs)

            code = '{code}\n{new}'.format( code = code,
                                           new  = decs )
        # ...

        # ...
        if expr.body:
            body = '\n'.join(self._print(i) for i in expr.body)

            code = '{code}\n{new}'.format( code = code,
                                           new  = body )
        # ...

        return code


def pycode(expr, **settings):
    """ Converts an expr to a string of Python code
    Parameters
    ==========
    expr : Expr
        A SymPy expression.
    fully_qualified_modules : bool
        Whether or not to write out full module names of functions
        (``math.sin`` vs. ``sin``). default: ``True``.
    enable_dependencies: bool
        Whether or not to print dependencies too (EvalField, Kernel, etc)
    Examples
    ========
    >>> from sympy import tan, Symbol
    >>> from sympy.printing.pycode import pycode
    >>> pycode(tan(Symbol('x')) + 1)
    'math.tan(x) + 1'
    """
    return PythonCodePrinter(settings).doprint(expr)
