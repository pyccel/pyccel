# coding: utf-8

from sympy import sympify, simplify, Symbol, Integer, Float, Add, Mul
from sympy import Piecewise, log
from sympy.abc import x

from sympy.core.basic import Basic
from sympy.core.expr import Expr, AtomicExpr
from sympy.core.compatibility import string_types
from sympy.core.operations import LatticeOp
from sympy.core.function import Derivative
from sympy.core.function import _coeff_isneg
from sympy.core.singleton import S
from sympy.utilities.iterables import iterable

from pyccel.parser.syntax.core import (AssignStmt, MultiAssignStmt, \
                                       IfStmt, ForStmt,WhileStmt)

from pyccel.ast.core import (For, Assign, Declare, Variable, \
                             datatype, While, NativeFloat, \
                             EqualityStmt, NotequalStmt, \
                             MultiAssign, \
                             FunctionDef, Import, Print, \
                             Comment, AnnotatedComment, \
                             IndexedVariable, Slice, If, \
                             Stencil, \
                             Zeros, Ones, Array, Len, Dot, IndexedElement)

from pyccel.complexity.basic import Complexity

__all__ = ["count_ops", "OpComplexity"]

# ...
def count_ops(expr, visual=True):
    """
    Return a representation (integer or expression) of the operations in expr.

    the number of each type of operation is shown
    with the core class types (or their virtual equivalent) multiplied by the
    number of times they occur.

    If expr is an iterable, the sum of the op counts of the
    items will be returned.
    """
    from sympy import Integral, Symbol
    from sympy.simplify.radsimp import fraction
    from sympy.logic.boolalg import BooleanFunction

    if not isinstance(expr, (Assign, For)):
        expr = sympify(expr)

    if isinstance(expr, Expr):

        ops = []
        args = [expr]
        NEG = Symbol('NEG')
        DIV = Symbol('DIV')
        SUB = Symbol('SUB')
        ADD = Symbol('ADD')
        while args:
            a = args.pop()

            # XXX: This is a hack to support non-Basic args
            if isinstance(a, string_types):
                continue

            if a.is_Rational:
                #-1/3 = NEG + DIV
                if a is not S.One:
                    if a.p < 0:
                        ops.append(NEG)
                    if a.q != 1:
                        ops.append(DIV)
                    continue
            elif a.is_Mul:
                if _coeff_isneg(a):
                    ops.append(NEG)
                    if a.args[0] is S.NegativeOne:
                        a = a.as_two_terms()[1]
                    else:
                        a = -a
                n, d = fraction(a)
                if n.is_Integer:
                    ops.append(DIV)
                    if n < 0:
                        ops.append(NEG)
                    args.append(d)
                    continue  # won't be -Mul but could be Add
                elif d is not S.One:
                    if not d.is_Integer:
                        args.append(d)
                    ops.append(DIV)
                    args.append(n)
                    continue  # could be -Mul
            elif a.is_Add:
                aargs = list(a.args)
                negs = 0
                for i, ai in enumerate(aargs):
                    if _coeff_isneg(ai):
                        negs += 1
                        args.append(-ai)
                        if i > 0:
                            ops.append(SUB)
                    else:
                        args.append(ai)
                        if i > 0:
                            ops.append(ADD)
                if negs == len(aargs):  # -x - y = NEG + SUB
                    ops.append(NEG)
                elif _coeff_isneg(aargs[0]):  # -x + y = SUB, but already recorded ADD
                    ops.append(SUB - ADD)
                continue
            if a.is_Pow and a.exp is S.NegativeOne:
                ops.append(DIV)
                args.append(a.base)  # won't be -Mul but could be Add
                continue
            if (a.is_Mul or
                a.is_Pow or
                a.is_Function or
                isinstance(a, Derivative) or
                    isinstance(a, Integral)):

                o = Symbol(a.func.__name__.upper())
                # count the args
                if (a.is_Mul or isinstance(a, LatticeOp)):
                    ops.append(o*(len(a.args) - 1))
                else:
                    ops.append(o)
            if (not a.is_Symbol) and (not isinstance(a, IndexedElement)):
                args.extend(a.args)

    elif type(expr) is dict:
        ops = [count_ops(k, visual=visual) +
               count_ops(v, visual=visual) for k, v in list(expr.items())]
    elif iterable(expr):
        ops = [count_ops(i, visual=visual) for i in expr]
    elif isinstance(expr, BooleanFunction):
        ops = []
        for arg in expr.args:
            ops.append(count_ops(arg, visual=True))
        o = Symbol(expr.func.__name__.upper())
        ops.append(o)
    elif isinstance(expr, Assign):
        ops = [count_ops(expr.rhs, visual=visual)]
    elif isinstance(expr, For):
        b = expr.iterable.args[0]
        e = expr.iterable.args[1]
        ops = [count_ops(i, visual=visual) for i in expr.body]
        ops = [i * (e-b) for i in ops]
    elif isinstance(expr, (Zeros, Ones)):
        ops = []
    elif not isinstance(expr, Basic):
        ops = []
    else:  # it's Basic not isinstance(expr, Expr):
        if not isinstance(expr, Basic):
            raise TypeError("Invalid type of expr")
        else:
            ops = []
            args = [expr]
            while args:
                a = args.pop()

                # XXX: This is a hack to support non-Basic args
                if isinstance(a, string_types):
                    continue

                if a.args:
                    o = Symbol(a.func.__name__.upper())
                    if a.is_Boolean:
                        ops.append(o*(len(a.args)-1))
                    else:
                        ops.append(o)
                    args.extend(a.args)

    if not ops:
        return S.Zero

    ops = simplify(Add(*ops))

    return ops
# ...

# ...
class OpComplexity(Complexity):
    """class for Operation complexity computation."""

    def cost(self, verbose=False):
        """
        Computes the complexity of the given code.

        verbose: bool
            talk more
        """
        # ...
        f = S.Zero
        for stmt in self.ast.statements:
            if isinstance(stmt, (AssignStmt, ForStmt)):
                f += count_ops(stmt.expr, visual=True)
        # ...

        # ...
        f = simplify(f)
        # ...

        # ...
        if verbose:
            print((" arithmetic cost         ~ " + str(f)))
        # ...

        return f
# ...

##############################################
if __name__ == "__main__":
    code = '''
n = 10

for i in range(0,n):
    for j in range(0,n):
        x = pow(i,2) + pow(i,3) + 3*i
        y = x / 3 + 2* x
    '''

    complexity = OpComplexity(code)
    print((complexity.cost()))
