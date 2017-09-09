# coding: utf-8

from sympy import sympify, simplify, Symbol, Integer, Float, Add, Mul
from sympy import Piecewise, log
from sympy.abc import x
from sympy import preorder_traversal
from sympy.core.expr import Expr, AtomicExpr
from sympy.core.singleton import S
from sympy.tensor.indexed import Idx
from sympy import limit, oo

from pyccel.parser  import PyccelParser
from pyccel.syntax import ( \
                           # statements
                           AssignStmt, MultiAssignStmt, \
                           IfStmt, ForStmt,WhileStmt \
                           )

from pyccel.types.ast import (For, Assign, Declare, Variable, \
                              datatype, While, NativeFloat, \
                              EqualityStmt, NotequalStmt, \
                              Argument, InArgument, InOutArgument, \
                              MultiAssign, OutArgument, Result, \
                              FunctionDef, Import, Print, \
                              Comment, AnnotatedComment, \
                              IndexedVariable, Slice, If, \
                              ThreadID, ThreadsNumber, \
                              Rational, NumpyZeros, NumpyLinspace, \
                              Stencil,ceil, \
                              NumpyOnes, NumpyArray, LEN, Dot, Min, Max,IndexedElement)

from pyccel.complexity.basic     import Complexity
from pyccel.complexity.operation import count_ops

__all__ = ["count_access", "MemComplexity"]

# ...
def count_access(expr, visual=True, local_vars=[]):
    """
    """
    from sympy import Integral, Symbol
    from sympy.simplify.radsimp import fraction
    from sympy.logic.boolalg import BooleanFunction

    if not isinstance(expr, (Assign, For)):
        expr = sympify(expr)

    ops   = []
    WRITE = Symbol('WRITE')
    READ  = Symbol('READ')

    local_vars = [str(a) for a in local_vars]

    if isinstance(expr, Expr):
        indices = []
        for arg in preorder_traversal(expr):
            if isinstance(arg, Idx):
                for a in arg.free_symbols:
                    indices.append(str(a))

        atoms = expr.atoms(Symbol)
        atoms = [str(i) for i in atoms]

        atoms      = set(atoms)
        indices    = set(indices)
        local_vars = set(local_vars)
        ignored = indices.union(local_vars)
        atoms = atoms - ignored
        ops = [READ]*len(atoms)
    elif isinstance(expr, Assign):
        if isinstance(expr.lhs, IndexedElement):
            name = str(expr.lhs.base)
        else:
            name = str(expr.lhs)
        ops  = [count_access(expr.rhs, visual=visual, local_vars=local_vars)]
        if not name in local_vars:
            ops += [WRITE]
    elif isinstance(expr, For):
        b = expr.iterable.args[0]
        e = expr.iterable.args[1]
        if isinstance(b, Symbol):
            local_vars.append(b)
        if isinstance(e, Symbol):
            local_vars.append(e)
        ops = [count_access(i, visual=visual, local_vars=local_vars) for i in expr.body]
        ops = [i * (e-b) for i in ops]
    elif isinstance(expr, (NumpyZeros, NumpyOnes)):
        ops = []

    if not ops:
        return S.Zero

    ops = simplify(Add(*ops))

    return ops
# ...

# ...
def free_parameters(expr):
    """
    """
    args = []
    if isinstance(expr, For):
        b = expr.iterable.args[0]
        e = expr.iterable.args[1]
        if isinstance(b, Symbol):
            args.append(str(b))
        if isinstance(e, Symbol):
            args.append(str(e))
        for i in expr.body:
            args += free_parameters(i)
        args = set(args)
        args = list(args)

    return args
# ...

# ...
from sympy import Poly, LM
def leading_term(expr, *args):

    expr = sympify(str(expr))
    P = Poly(expr, *args)
    d = P.as_dict()
    degree = P.total_degree()
    for key, value in d.items():
        if sum(key) == degree:
            return value * LM(P)
    return 0
# ...

# ...
class MemComplexity(Complexity):
    """Abstract class for complexity computation."""

    @property
    def free_parameters(self):
        # ...
        args = []
        for stmt in self.ast.statements:
            if isinstance(stmt, ForStmt):
                args += free_parameters(stmt.expr)
        # ...
        args = [Symbol(i) for i in args]
        return args

    def cost(self, local_vars=[]):
        """Computes the complexity of the given code."""
        # ...
        m = S.Zero
        f = S.Zero
        # ...

        # ...
        for stmt in self.ast.statements:
            if isinstance(stmt, (AssignStmt, ForStmt)):
                m += count_access(stmt.expr, visual=True, local_vars=local_vars)
                f += count_ops(stmt.expr, visual=True)
        # ...

        # ...
        m = simplify(m)
        f = simplify(f)
        # ...

        # ...
        d = {}
        d['m'] = m
        d['f'] = f
        # ...

        return d

    def intensity(self, d=None, args=None):
        # ...
        if d is None:
            d = self.cost(local_vars=['r', 'u', 'v'])
        # ...

        # ...
        if args is None:
            args = self.free_parameters
        # ...

        # ...
        f = d['f']
        m = d['m']
        lt_f = leading_term(f, *args)
        lt_m = leading_term(m, *args)
        # ...

        return lt_f/lt_m
# ...

##############################################
if __name__ == "__main__":
    import sys
    filename = sys.argv[1]

    M = MemComplexity(filename)
    d = M.cost(local_vars=['r', 'u', 'v'])

    f = d['f']
    m = d['m']
    print "f = ", f
    print "m = ", m

    q = M.intensity()
    print ">>> computational intensity ~", q

#    # ... computational intensity
#    q = f / m
#    q = simplify(q)
#    t_f = Symbol('t_f')
#    t_m = Symbol('t_m')
#    c = f * t_f + m * t_m
#    # ...

