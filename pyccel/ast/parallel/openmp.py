# coding: utf-8


from sympy.core import Tuple
from sympy.utilities.iterables import iterable

from pyccel.ast.core import DottedName
from pyccel.ast.core import Variable, IndexedVariable, IndexedElement
from pyccel.ast.core import Assign, Declare, AugAssign
from pyccel.ast.core import NativeBool, NativeFloat
from pyccel.ast.core import NativeComplex, NativeDouble, NativeInteger
from pyccel.ast.core import DataType
from pyccel.ast.core import DataTypeFactory
from pyccel.ast.core import Block, ParallelBlock
from pyccel.ast.core import Range, Tile, Tensor
from pyccel.ast.core import Zeros
from pyccel.ast.core import Ones
from pyccel.ast.core import Comment
from pyccel.ast.core import AnnotatedComment
from pyccel.ast.core import EmptyLine
from pyccel.ast.core import Print
from pyccel.ast.core import Len
from pyccel.ast.core import Import
from pyccel.ast.core import For, While, If, Del, Sync, With
from pyccel.ast.core import FunctionDef, ClassDef
from pyccel.ast.core import MethodCall, FunctionCall

from pyccel.ast.parallel.basic import Basic

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
class OMP_Parallel(ParallelBlock, OMP):
    """
    OMP Parallel construct statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_Parallel
    >>> from pyccel.parallel.openmp import OMP_ParallelNumThreadClause
    >>> from pyccel.parallel.openmp import OMP_ParallelDefaultClause
    >>> from pyccel.ast.core import Variable, Assign, Block
    >>> n = Variable('int', 'n')
    >>> x = Variable('int', 'x')
    >>> body = [Assign(x,2.*n + 1.), Assign(n, n + 1)]
    >>> variables = [x,n]
    >>> clauses = [OMP_ParallelNumThreadClause(4), OMP_ParallelDefaultClause('shared')]
    >>> OMP_Parallel(clauses, variables, body)
    #pragma parallel num_threads(4) default(shared)
    x := 1.0 + 2.0*n
    n := 1 + n
    """
    _prefix = '#pragma'
    def __new__(cls, clauses, variables, body):
        if not iterable(clauses):
            raise TypeError('Expecting an iterable for clauses')

        _valid_clauses = (OMP_ParallelNumThreadClause, \
                          OMP_ParallelDefaultClause, \
                          OMP_PrivateClause, \
                          OMP_SharedClause, \
                          OMP_FirstPrivateClause, \
                          OMP_CopyinClause, \
                          OMP_ReductionClause, \
                          OMP_ParallelProcBindClause)

        for clause in clauses:
            if not isinstance(clause, _valid_clauses):
                raise TypeError('Wrong clause for OMP_Parallel')

        return ParallelBlock.__new__(cls, clauses, variables, body)

class OMP_ParallelNumThreadClause(OMP):
    """
    OMP ParallelNumThreadClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_ParallelNumThreadClause
    >>> OMP_ParallelNumThreadClause(4)
    num_threads(4)
    """
    def __new__(cls, *args, **options):
        num_threads = args[0]
        return Basic.__new__(cls, num_threads)

    @property
    def num_threads(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        return 'num_threads({})'.format(sstr(self.num_threads))

class OMP_ParallelDefaultClause(OMP):
    """
    OMP ParallelDefaultClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_ParallelDefaultClause
    >>> OMP_ParallelDefaultClause('shared')
    default(shared)
    """
    def __new__(cls, *args, **options):
        status = args[0]
        return Basic.__new__(cls, status)

    @property
    def status(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        status = self.status
        if status:
            status = sstr(self.status)
        else:
            status = ''
        return 'default({})'.format(status)

class OMP_ParallelProcBindClause(OMP):
    """
    OMP ParallelProcBindClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_ParallelProcBindClause
    >>> OMP_ParallelProcBindClause('master')
    proc_bind(master)
    """
    def __new__(cls, *args, **options):
        status = args[0]
        return Basic.__new__(cls, status)

    @property
    def status(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        status = self.status
        if status:
            status = sstr(self.status)
        else:
            status = ''
        return 'proc_bind({})'.format(status)

class OMP_PrivateClause(OMP):
    """
    OMP PrivateClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_PrivateClause
    >>> OMP_PrivateClause('x', 'y')
    private(x, y)
    """
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'private({})'.format(args)

class OMP_SharedClause(OMP):
    """
    OMP SharedClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_SharedClause
    >>> OMP_SharedClause('x', 'y')
    shared(x, y)
    """
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'shared({})'.format(args)

class OMP_FirstPrivateClause(OMP):
    """
    OMP FirstPrivateClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_FirstPrivateClause
    >>> OMP_FirstPrivateClause('x', 'y')
    firstprivate(x, y)
    """
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'firstprivate({})'.format(args)

class OMP_LastPrivateClause(OMP):
    """
    OMP LastPrivateClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_LastPrivateClause
    >>> OMP_LastPrivateClause('x', 'y')
    lastprivate(x, y)
    """
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'lastprivate({})'.format(args)

class OMP_CopyinClause(OMP):
    """
    OMP CopyinClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_CopyinClause
    >>> OMP_CopyinClause('x', 'y')
    copyin(x, y)
    """
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'copyin({})'.format(args)

class OMP_ReductionClause(OMP):
    """
    OMP ReductionClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_ReductionClause
    >>> OMP_ReductionClause('+', 'x', 'y')
    reduction('+': (x, y))
    """
    def __new__(cls, *args, **options):
        op = args[0]
        arguments = args[1:]
        return Basic.__new__(cls, op, arguments)

    @property
    def operation(self):
        return self._args[0]

    @property
    def variables(self):
        return self._args[1:]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        op   = sstr(self.operation)
        return "reduction('{0}': {1})".format(op, args)

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
        clauses   = stmt.clauses

        return OMP_Parallel(clauses, variables, body)

    return stmt
##########################################################
