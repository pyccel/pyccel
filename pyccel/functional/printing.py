from sympy.core import Symbol
from sympy import Tuple

from pyccel.codegen.printing.pycode import PythonCodePrinter as PyccelPythonCodePrinter

class PythonCodePrinter(PyccelPythonCodePrinter):

    def __init__(self, settings=None):
        PyccelPythonCodePrinter.__init__(self, settings=settings)

    def _print_FunctionalMap(self, expr):
        allocations = '\n'.join(self._print(i) for i in expr.allocations)
        inits       = '\n'.join(self._print(i) for i in expr.inits)
        decs        = '\n'.join(self._print(i) for i in expr.decs)
        stmts       = '\n'.join(self._print(i) for i in expr.stmts)
        results     = '\n'.join(self._print(i) for i in expr.results)

        code = '{allocations}\n{inits}\n{decs}\n{stmts}\n{results}'
        code = code.format( allocations = allocations,
                            inits       = inits,
                            decs        = decs,
                            stmts       = stmts,
                            results     = results )

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
