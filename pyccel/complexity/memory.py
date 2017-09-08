# coding: utf-8

from sympy import sympify, simplify, Symbol, Integer, Float, Add, Mul
from sympy import Piecewise, log
from sympy.abc import x

from pyccel.parser  import PyccelParser
from pyccel.syntax import ( \
                           # statements
                           AssignStmt, MultiAssignStmt, \
                           IfStmt, ForStmt,WhileStmt \
                           )

from pyccel.types.ast import (Assign, For)

from pyccel.complexity.basic import Complexity

__all__ = ["count_mem", "MemComplexity"]

# ...
def count_mem(expr, visual=True):
    """
    """
    return 0
# ...

# ...
class MemComplexity(Complexity):
    """Abstract class for complexity computation."""

    def cost(self):
        """Computes the complexity of the given code."""
        # ...
        cost = 0
        for stmt in self.ast.statements:
            if isinstance(stmt, (AssignStmt, ForStmt)):
                cost += count_mem(stmt.expr)
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
    complexity = MemComplexity(filename)
    print complexity.cost()

