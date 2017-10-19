# coding: utf-8


from sympy.core import Tuple
from sympy.utilities.iterables import iterable

from pyccel.types.ast import DottedName
from pyccel.types.ast import Variable, IndexedVariable, IndexedElement
from pyccel.types.ast import Assign, Declare, AugAssign
from pyccel.types.ast import NativeBool, NativeFloat
from pyccel.types.ast import NativeComplex, NativeDouble, NativeInteger
from pyccel.types.ast import DataType
from pyccel.types.ast import DataTypeFactory
from pyccel.types.ast import Block, ParallelBlock
from pyccel.types.ast import Range, Tile, Tensor
from pyccel.types.ast import Zeros
from pyccel.types.ast import Ones
from pyccel.types.ast import Comment
from pyccel.types.ast import AnnotatedComment
from pyccel.types.ast import EmptyLine
from pyccel.types.ast import Print
from pyccel.types.ast import Len
from pyccel.types.ast import Import

from pyccel.types.ast import For, While, If, Del, Sync, With
from pyccel.types.ast import FunctionDef, ClassDef
from pyccel.types.ast import MethodCall, FunctionCall

from pyccel.parallel.basic import Basic

##########################################################
#               Base class for OpenMP
##########################################################
class OMP(Basic):
    """Base class for OpenMP."""
    pass
##########################################################

##########################################################
#                 Basic Statements
##########################################################
class OMP_Parallel(AnnotatedComment, OMP):
    """OMP Parallel statement."""
    def __new__(cls, *args, **options):
        clauses = args[0]

#        valid_clauses = (ParallelNumThreadClause, \
#                         ParallelDefaultClause, \
#                         PrivateClause, \
#                         SharedClause, \
#                         FirstPrivateClause, \
#                         CopyinClause, \
#                         ReductionClause, \
#                         ParallelProcBindClause)
#
#        txt = 'parallel'
#        for clause in self.clauses:
#            if isinstance(clause, valid_clauses):
#                txt = '{0} {1}'.format(txt, clause.expr)
#            else:
#                raise TypeError('Wrong clause for ParallelStmt')

        txt = 'parallel'
        return AnnotatedComment.__new__(cls, 'omp', txt)

class OMP_EndConstruct(AnnotatedComment, OMP):
    """OMP EndConstruct statement."""
    def __new__(cls, *args, **options):
        construct = 'parallel'
        simd      = ''
        nowait    = ''
        txt = 'end {0} {1} {2}'.format(construct, \
                                       simd, \
                                       nowait)
        return AnnotatedComment.__new__(cls, 'omp', txt)

##########################################################


##########################################################
#             useful functions
##########################################################
def openmpfy(stmt, **options):
    """
    Converts some statements to OpenMP statments.

    stmt: stmt, list
        statement or a list of statements
    """
    if isinstance(stmt, (list, tuple, Tuple)):
        return [openmpfy(i, **options) for i in stmt]
#    if isinstance(stmt, OPENMP):
#        return stmt
    if isinstance(stmt, Tensor):
        raise NotImplementedError('Tensor stmt not available')
    if isinstance(stmt, For):
        iterable = openmpfy(stmt.iterable, **options)
        target   = stmt.target
        body     = openmpfy(stmt.body, **options)
        return For(target, iterable, body, strict=False)
    if isinstance(stmt, list):
        return [openmpfy(a, **options) for a in stmt]
    if isinstance(stmt, While):
        test = openmpfy(stmt.test, **options)
        body = openmpfy(stmt.body, **options)
        return While(test, body)
    if isinstance(stmt, If):
        args = []
        for block in stmt.args:
            test  = block[0]
            stmts = block[1]
            t = openmpfy(test,  **options)
            s = openmpfy(stmts, **options)
            args.append((t,s))
        return If(*args)
    if isinstance(stmt, FunctionDef):
        name        = openmpfy(stmt.name,        **options)
        arguments   = openmpfy(stmt.arguments,   **options)
        results     = openmpfy(stmt.results,     **options)
        body        = openmpfy(stmt.body,        **options)
        local_vars  = openmpfy(stmt.local_vars,  **options)
        global_vars = openmpfy(stmt.global_vars, **options)

        return FunctionDef(name, arguments, results, \
                           body, local_vars, global_vars)
    if isinstance(stmt, ClassDef):
        name        = openmpfy(stmt.name,        **options)
        attributs   = openmpfy(stmt.attributs,   **options)
        methods     = openmpfy(stmt.methods,     **options)
        options     = openmpfy(stmt.options,     **options)

        return ClassDef(name, attributs, methods, options)
    if isinstance(stmt, Sync):
        raise NotImplementedError('Sync stmt not available')
    if isinstance(stmt, With):
        raise NotImplementedError('With stmt not available')
    if isinstance(stmt, ParallelBlock):
        variables = stmt.variables
        body      = stmt.body
        clauses   = []

        prelude  = [OMP_Parallel(clauses)]
        epilog   = [OMP_EndConstruct()]
        body     = prelude + body + epilog

        return Block(variables, body)

    return stmt
##########################################################
