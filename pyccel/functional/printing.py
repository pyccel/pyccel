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

    def _print_BasicBlock(self, expr):
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

    # .....................................................
    #                   OpenMP statements
    # .....................................................
    def _print_OMP_Parallel(self, expr):
        clauses = ' '.join(self._print(i)  for i in expr.clauses)
        body    = '\n'.join(self._print(i) for i in expr.body)

        # ... TODO adapt get_statement to have continuation with OpenMP
        prolog = '#$ omp parallel {clauses}\n'.format(clauses=clauses)
        epilog = '#$ omp end parallel\n'
        # ...

        # ...
        code = ('{prolog}'
                '{body}\n'
                '{epilog}').format(prolog=prolog, body=body, epilog=epilog)
        # ...

        return self._get_statement(code)

    def _print_OMP_For(self, expr):
        # ...
        loop    = self._print(expr.loop)
        clauses = ' '.join(self._print(i)  for i in expr.clauses)

        nowait  = ''
        if not(expr.nowait is None):
            nowait = 'nowait'
        # ...

        # ... TODO adapt get_statement to have continuation with OpenMP
        prolog = '#$ omp do {clauses}\n'.format(clauses=clauses)
        epilog = '#$ omp end do {0}\n'.format(nowait)
        # ...

        # ...
        code = ('{prolog}'
                '{loop}\n'
                '{epilog}').format(prolog=prolog, loop=loop, epilog=epilog)
        # ...

        return self._get_statement(code)

    def _print_OMP_NumThread(self, expr):
        return 'num_threads({})'.format(self._print(expr.num_threads))

    def _print_OMP_Default(self, expr):
        status = expr.status
        if status:
            status = self._print(expr.status)
        else:
            status = ''
        return 'default({})'.format(status)

    def _print_OMP_ProcBind(self, expr):
        status = expr.status
        if status:
            status = self._print(expr.status)
        else:
            status = ''
        return 'proc_bind({})'.format(status)

    def _print_OMP_Private(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'private({})'.format(args)

    def _print_OMP_Shared(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'shared({})'.format(args)

    def _print_OMP_FirstPrivate(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'firstprivate({})'.format(args)

    def _print_OMP_LastPrivate(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'lastprivate({})'.format(args)

    def _print_OMP_Copyin(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        return 'copyin({})'.format(args)

    def _print_OMP_Reduction(self, expr):
        args = ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        op   = self._print(expr.operation)
        return "reduction({0}: {1})".format(op, args)

    def _print_OMP_Schedule(self, expr):
        kind = self._print(expr.kind)

        chunk_size = ''
        if expr.chunk_size:
            chunk_size = ', {0}'.format(self._print(expr.chunk_size))

        return 'schedule({0}{1})'.format(kind, chunk_size)

    def _print_OMP_Ordered(self, expr):
        n_loops = ''
        if expr.n_loops:
            n_loops = '({0})'.format(self._print(expr.n_loops))

        return 'ordered{0}'.format(n_loops)

    def _print_OMP_Collapse(self, expr):
        n_loops = '{0}'.format(self._print(expr.n_loops))

        return 'collapse({0})'.format(n_loops)

    def _print_OMP_Linear(self, expr):
        variables= ', '.join('{0}'.format(self._print(i)) for i in expr.variables)
        step = self._print(expr.step)
        return "linear({0}: {1})".format(variables, step)

    def _print_OMP_If(self, expr):
        return 'if({})'.format(self._print(expr.test))
    # .....................................................



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
