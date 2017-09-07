# coding: utf-8

from sympy import sympify, Symbol, Integer, Float, Add, Mul
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

from pyccel.parser  import PyccelParser
from pyccel.syntax import ( \
                           # statements
                           AssignStmt, MultiAssignStmt, \
                           IfStmt, ForStmt,WhileStmt \
                           )

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
            if not a.is_Symbol:
                args.extend(a.args)

    elif type(expr) is dict:
        ops = [count_ops(k, visual=visual) +
               count_ops(v, visual=visual) for k, v in expr.items()]
    elif iterable(expr):
        ops = [count_ops(i, visual=visual) for i in expr]
    elif isinstance(expr, BooleanFunction):
        ops = []
        for arg in expr.args:
            ops.append(count_ops(arg, visual=True))
        o = Symbol(expr.func.__name__.upper())
        ops.append(o)
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

    ops = Add(*ops)

    return ops
# ...

# ...
class Complexity(object):
    """Abstract class for complexity computation."""
    def __init__(self, filename):
        """Constructor for the Codegen class.

        body: list
            list of statements.
        """
        # ... TODO improve once TextX will handle indentation
        from pyccel.codegen import clean, preprocess, make_tmp_file

        clean(filename)

        filename_tmp = make_tmp_file(filename)
        preprocess(filename, filename_tmp)
        filename = filename_tmp
        # ...

        self._filename = filename

    @property
    def filename(self):
        """Returns the name of the file to convert."""
        return self._filename

    def cost(self):
        """Computes the complexity of the given code."""
        # ...
        filename = self.filename
        # ...

        # ...
        pyccel = PyccelParser()
        ast    = pyccel.parse_from_file(filename)
        # ...

        # ...
        cost = 0
        for stmt in ast.statements:
            if isinstance(stmt, AssignStmt):
                cost += count_ops(stmt.expr.rhs)
#            elif isinstance(stmt, MultiAssignStmt):
#                pass
            elif isinstance(stmt, ForStmt):
                expr = stmt.expr
                b = expr.iterable.args[0]
                e = expr.iterable.args[1]
                cost += count_ops(stmt.body.expr) * (e-b)
            elif isinstance(stmt,WhileStmt):
                pass
            elif isinstance(stmt, IfStmt):
                pass
            else:
                pass
        # ...

        return cost
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
    complexity = Complexity(filename)
    print complexity.cost()

