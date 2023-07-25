#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
from sympy import cse as sympy_cse
from sympy import Sum
from sympy import Tuple

from sympy.core.function  import Function
from pyccel.ast.builtins import Lambda
from pyccel.ast.core import Import
from pyccel.ast.core import Return, FunctionDef
from pyccel.ast.core import Assign, create_variable
from pyccel.ast.core import AugAssign
from pyccel.ast.core import For
from pyccel.ast.internals      import PyccelSymbol
from pyccel.ast.functionalexpr import GeneratorComprehension as GC
from pyccel.ast.functionalexpr import FunctionalSum


def cse(expr):
    """ symplify a complicated sympy expression
        into a list of expression using the cse
        sympy function
    """
    ls = list(expr.atoms(Sum))
    if not ls:
        return [expr]
    ls += [expr]
    (ls, _) = sympy_cse(ls)

    (vars_old, stmts) = map(list, zip(*ls))
    vars_new = []
    free_gl = expr.free_symbols
    #free_gl.update(expr.atoms(IndexedBase)) #What should this be instead?
    free_gl.update(vars_old)
    stmts.append(expr)

    for i in range(len(stmts) - 1):
        free = stmts[i].free_symbols
        free = free.difference(free_gl)
        free = list(free)
        var = create_variable(stmts[i])
        if len(free) > 0:
            var = var[free]
        vars_new.append(var)
    for i in range(len(stmts) - 1):
        stmts[i + 1] = stmts[i + 1].replace(vars_old[i],
                vars_new[i])
        stmts[-1] = stmts[-1].replace(stmts[i], vars_new[i])

    allocate = []
    for i in range(len(stmts) - 1):
        stmts[i] = Assign(vars_new[i], stmts[i])
        stmts[i] = pyccel_sum(stmts[i])
        if isinstance(vars_new[i], IndexedElement):
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
            var = PyccelSymbol(vars_new[i].base)
            stmt = Assign(var, Function('empty')(size[0]))
            allocate.append(stmt)
            stmts[i] = For(ind[0], Function('range')(size[0]), [stmts[i]])
    lhs = create_variable(expr)
    stmts[-1] = Assign(lhs, stmts[-1])
    imports = [Import('empty', 'numpy')]
    return imports + allocate + stmts


def pyccel_sum(expr):
    """ convert the sympy sum to the
        pyccel node FunctionalSum
    """
    if not(isinstance(expr, Assign) and isinstance(expr.rhs, Sum)):
        return expr
    lhs = expr.lhs
    expr = expr.rhs
    index = expr.args[1]
    target = Function('range')(index[1], index[2])
    body = AugAssign(lhs, '+', expr.args[0])
    stmt = For(index[0], target, [body])
    stmt = FunctionalSum([stmt], expr.args[0], lhs)

    return stmt


def lambdify(expr, args):
    if isinstance(args, Lambda):
        new_expr = args.expr
        new_expr = Return(new_expr)
        new_expr.ast = expr
        f_arguments = args.variables
        func = FunctionDef('lambda', f_arguments, [], [new_expr])
        return func


    code = compile(args.body[0],'','single')
    g={}
    eval(code,g)
    f_name = str(args.name)
    code = g[f_name]
    new_args = args.arguments
    new_expr = code(*new_args)
    f_arguments = list(new_expr.free_symbols)
    stmts = cse(new_expr)
    if isinstance(stmts[-1], (Assign, GC)):
        var = stmts[-1].lhs
    else:
        var  = create_variable(expr)
        stmts[-1] = Assign(var, stmts[-1])
    stmts += [Return([var])]
    set_fst(stmts, args.ast)
    func = FunctionDef(f_name, new_args, [], stmts ,decorators = args.decorators)
    return func

def set_fst(expr, fst):
    if isinstance(expr, (tuple,list)):
        for i in expr:set_fst(i, fst)
    elif isinstance(expr, For):
        set_fst(expr.body, fst)
    elif isinstance(expr, (Assign, AugAssign)):
        expr.ast = fst
    elif isinstance(expr, GC):
        expr.ast = fst
        set_fst(expr.loops, fst)

