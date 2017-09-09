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

__all__ = ["count_mem", "MemComplexity"]

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
                indices.append(str(arg))

        atoms = expr.atoms(Symbol)
        atoms = [str(i) for i in atoms]

        atoms      = set(atoms)
        indices    = set(indices)
        local_vars = set(local_vars)
        ignored = indices.union(local_vars)
        atoms = atoms - ignored
#        print type(expr), expr
#        print "indices : ", indices
#        print "atoms : ", atoms
#        print "ignored : ", ignored
        ops = [READ]*len(atoms)
    elif isinstance(expr, Assign):
        if isinstance(expr.lhs, IndexedElement):
            name = str(expr.lhs.base)
        else:
            name = str(expr.lhs)
#        print "type(expr.rhs) = ", type(expr.rhs)
        ops  = [count_access(expr.rhs, visual=visual, local_vars=local_vars)]
        if not Symbol(name) in local_vars:
            ops += [WRITE]
    elif isinstance(expr, For):
        b = expr.iterable.args[0]
        e = expr.iterable.args[1]
        if isinstance(b, Symbol):
            local_vars.append(b)
        if isinstance(e, Symbol):
            local_vars.append(e)
        ops = [count_access(i, visual=visual, local_vars=local_vars) for i in expr.body]
#        print ">>> ops = ", ops
        ops = [i * (e-b) for i in ops]
    elif isinstance(expr, (NumpyZeros, NumpyOnes)):
        ops = []

    if not ops:
        return S.Zero

    ops = simplify(Add(*ops))

    return ops
# ...

# ...
def count_mem(expr, visual=True, local_vars=[]):
    """
    """
    f = count_ops(expr, visual=True)
    m = count_access(expr, visual=True, local_vars=local_vars)

#    t_f = Symbol('t_f')
#    t_m = Symbol('t_m')
#    return f * t_f + m * t_m
# ...

# ...
class MemComplexity(Complexity):
    """Abstract class for complexity computation."""

    def cost(self, local_vars=[]):
        """Computes the complexity of the given code."""
        # ...
        f = S.Zero
        m = S.Zero
        for stmt in self.ast.statements:
            if isinstance(stmt, (AssignStmt, ForStmt)):
                f += count_ops(stmt.expr, visual=True)
                m += count_access(stmt.expr, visual=True, local_vars=local_vars)
        # ...

        print "f: ", f
        print "m: ", m

        return f,m
# ...

##############################################
if __name__ == "__main__":
#    expr = sympify('(x+1)**2+x+1')
#    print expr
#    d = count_ops(expr)
#    print d
#
#    f = x**2
#    g = log(x)
#    expr = Piecewise( (0, x<-1), (f, x<=1), (g, True))
#    d = count_ops(expr)
#    print d

    import sys
    filename = sys.argv[1]
    complexity = MemComplexity(filename)
    complexity.cost(local_vars=['r', 'u', 'v'])
