# coding: utf-8

from itertools import groupby
import numpy as np

from sympy.core.symbol  import Symbol
from sympy.core.compatibility import with_metaclass
from sympy.core.singleton import Singleton
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse
from sympy.core import Tuple
from sympy.utilities.iterables import iterable
from sympy.core.function import Function
from sympy.core.function import UndefinedFunction

from pyccel.ast.core import Module, Program
from pyccel.ast.core import DottedName
from pyccel.ast.core import Variable, IndexedVariable, IndexedElement
from pyccel.ast.core import Assign, Declare, AugAssign
from pyccel.ast.core import Block
from pyccel.ast.core import Tensor
from pyccel.ast.core import Comment
from pyccel.ast.core import EmptyLine
from pyccel.ast.core import Import
from pyccel.ast.core import For, ForIterator, While, If, Del
from pyccel.ast.core import FunctionDef, ClassDef
from pyccel.ast.numpyext import Zeros, Ones

from pyccel.ast.parallel.basic        import Basic
from pyccel.ast.parallel.communicator import UniversalCommunicator

__all__ = (
    'MPI',
    'mpify'
)

##########################################################
#               Base class for MPI
##########################################################
class MPI(Basic):
    """Base class for MPI."""
    pass

##########################################################
#             useful functions
##########################################################
def mpify(stmt, **options):
    """
    Converts some statements to MPI statments.

    stmt: stmt, list
        statement or a list of statements
    """
    if isinstance(stmt, (list, tuple, Tuple)):
        return [mpify(i, **options) for i in stmt]

    if isinstance(stmt, MPI):
        return stmt

    if isinstance(stmt, Tensor):
        options['label'] = stmt.name
        return stmt

    if isinstance(stmt, ForIterator):
        iterable = mpify(stmt.iterable, **options)
        target   = stmt.target
        body     = mpify(stmt.body, **options)
        return ForIterator(target, iterable, body, strict=False)

    if isinstance(stmt, For):
        iterable = mpify(stmt.iterable, **options)
        target   = stmt.target
        body     = mpify(stmt.body, **options)
        return For(target, iterable, body, strict=False)

    if isinstance(stmt, list):
        return [mpify(a, **options) for a in stmt]

    if isinstance(stmt, While):
        test = mpify(stmt.test, **options)
        body = mpify(stmt.body, **options)
        return While(test, body)

    if isinstance(stmt, If):
        args = []
        for block in stmt.args:
            test  = block[0]
            stmts = block[1]
            t = mpify(test,  **options)
            s = mpify(stmts, **options)
            args.append((t,s))
        return If(*args)

    if isinstance(stmt, FunctionDef):
        return stmt
        # TODO uncomment this
#        name        = mpify(stmt.name,        **options)
#        arguments   = mpify(stmt.arguments,   **options)
#        results     = mpify(stmt.results,     **options)
#        body        = mpify(stmt.body,        **options)
#        local_vars  = mpify(stmt.local_vars,  **options)
#        global_vars = mpify(stmt.global_vars, **options)
#
#        return FunctionDef(name, arguments, results, \
#                           body, local_vars, global_vars)

    if isinstance(stmt, ClassDef):
        name        = mpify(stmt.name,        **options)
        attributs   = mpify(stmt.attributs,   **options)
        methods     = mpify(stmt.methods,     **options)
        options     = mpify(stmt.options,     **options)

        return ClassDef(name, attributs, methods, options)

    if isinstance(stmt, Assign):
        if isinstance(stmt.rhs, Tensor):
            lhs = stmt.lhs
            options['label'] = lhs.name
            rhs = mpify(stmt.rhs, **options)

            return Assign(lhs, rhs, \
                          strict=stmt.strict, \
                          status=stmt.status, \
                          like=stmt.like)

    if isinstance(stmt, Del):
        variables = [mpify(a, **options) for a in stmt.variables]
        return Del(variables)

    if isinstance(stmt, Ones):
        if stmt.grid:
            lhs   = stmt.lhs
            shape = stmt.shape
            grid  = mpify(stmt.grid, **options)
            return Ones(lhs, grid=grid)

    if isinstance(stmt, Zeros):
        if stmt.grid:
            lhs   = stmt.lhs
            shape = stmt.shape
            grid  = mpify(stmt.grid, **options)
            return Zeros(lhs, grid=grid)

    if isinstance(stmt, Module):
        name        = mpify(stmt.name,        **options)
        variables   = mpify(stmt.variables,   **options)
        funcs       = mpify(stmt.funcs    ,   **options)
        classes     = mpify(stmt.classes  ,   **options)
        imports     = mpify(stmt.imports  ,   **options)
        imports    += [Import('mpi')]
        # TODO add stdlib_parallel_mpi module

        return Module(name, variables, funcs, classes,
                      imports=imports)

    if isinstance(stmt, Program):
        name        = mpify(stmt.name,        **options)
        variables   = mpify(stmt.variables,   **options)
        funcs       = mpify(stmt.funcs    ,   **options)
        classes     = mpify(stmt.classes  ,   **options)
        imports     = mpify(stmt.imports  ,   **options)
        body        = mpify(stmt.body  ,   **options)
        modules     = mpify(stmt.modules  ,   **options)
        imports    += [Import('mpi')]
        # TODO improve this import, without writing 'mod_...'
        #      maybe we should create a new class for this import
        imports    += [Import('mod_pyccel_stdlib_parallel_mpi')]

        return Program(name, variables, funcs, classes, body,
                       imports=imports, modules=modules)

    return stmt
