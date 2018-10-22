
from sympy import cse
from sympy import Sum
from sympy import IndexedBase, Indexed
from sympy import KroneckerDelta, Heaviside
from sympy import Symbol, sympify, symbols
from sympy import Integer, Float
from sympy import true, false
from sympy import Tuple

from sympy.core.function  import Function
from pyccel.ast import Import, TupleImport
from pyccel.ast import Return
from pyccel.ast import Assign
from pyccel.ast import AugAssign, CodeBlock
from pyccel.ast import For, FunctionalFor, ForIterator
from pyccel.ast import GeneratorComprehension as GC
from pyccel.ast import FunctionalSum, FunctionalMax, FunctionalMin
from pyccel.ast import If, IfTernaryOperator
def f(expr):
    ls = list(expr.atoms(Sum))
    if not ls:
        return expr
    ls += [expr]
    (ls, m) = cse(ls)

    (vars_old, stmts) = map(list, zip(*ls))
    vars_new = []
    free_gl = expr.free_symbols
    free_gl.update(expr.atoms(IndexedBase))
    free_gl.update(vars_old)
    stmts.append(expr)

    for i in range(len(stmts) - 1):
        free = stmts[i].free_symbols
        free = free.difference(free_gl)
        free = list(free)
        var = create_variable(stmts[i])
        if len(free) > 0:
            var = IndexedBase(var)[free]
        vars_new.append(var)
    for i in range(len(stmts) - 1):
        stmts[i + 1] = stmts[i + 1].replace(vars_old[i],
                vars_new[i])
        stmts[-1] = stmts[-1].replace(stmts[i], vars_new[i])

    allocate = []
    for i in range(len(stmts) - 1):
        stmts[i] = Assign(vars_new[i], stmts[i])
        if isinstance(vars_new[i], Indexed):
            ind = vars_new[i].indices
            tp = list(stmts[i + 1].atoms(Tuple))
            size = None
            size = [None] * len(ind)
            for (j, k) in enumerate(ind):
                for t in tp:
                    if k == t[0]:
                        size[j] = t[2] - t[1] + 1
                        break
            if not all(size):
                raise ValueError('Unable to find range of index')
            name = _get_name(vars_new[i].base)
            var = Symbol(name)
            stmt = Assign(var, Function('zeros')(size[0]))
            allocate.append(stmt)
            stmts[i] = For(ind[0], Function('range')(size[0]), [stmts[i]], strict=False)

    stmts[-1] = Assign(lhs, stmts[-1])
    imports = Import('numpy','empty')
    return CodeBlock(allocate + stmts)
    

def g(expr):
    index = expr.args[1]
    target = Function('range')(index[1], index[2])
    body = AugAssign(lhs, '+', expr.args[0])
    body = self._annotate(body, **settings)
    stmt = For(index[0], target, [body], strict=False)
    stmt = FunctionalSum([stmt], body, [], None)
    return expr



