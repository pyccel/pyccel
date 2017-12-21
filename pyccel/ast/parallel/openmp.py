# coding: utf-8


from sympy.core import Tuple
from sympy.utilities.iterables import iterable
from sympy import Integer

from pyccel.ast.core import Nil
from pyccel.ast.core import get_initial_value
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
from pyccel.ast.core import For, ForIterator, While, If, Del, Sync, With
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

class OMP_For(ForIterator, OMP):
    """
    OMP Parallel For construct statement.

    Examples

    """
    _prefix = '#pragma'
    def __new__(cls, target, iterable, body, clauses, nowait):
        if not iterable(clauses):
            raise TypeError('Expecting an iterable for clauses')

        _valid_clauses = (OMP_ScheduleClause, \
                          OMP_PrivateClause, \
                          OMP_FirstPrivateClause, \
                          OMP_LastPrivateClause, \
                          OMP_ReductionClause, \
                          OMP_CollapseClause, \
                          OMP_OrderedClause, \
                          OMP_LinearClause)

        for clause in clauses:
            if not isinstance(clause, _valid_clauses):
                raise TypeError('Wrong clause for OMP_For, '
                               'given {0}'.format(type(clause)))

        return Basic.__new__(cls, target, iterable, body, clauses, nowait)

    @property
    def target(self):
        return self._args[0]

    @property
    def iterable(self):
        return self._args[1]

    @property
    def body(self):
        return self._args[2]

    @property
    def clauses(self):
        return self._args[3]

    @property
    def nowait(self):
        return self._args[4]


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
        return self._args[1]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        op   = sstr(self.operation)
        return "reduction('{0}': {1})".format(op, args)

class OMP_ScheduleClause(OMP):
    """
    OMP ScheduleClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_ScheduleClause
    >>> OMP_ScheduleClause('static', 2)
    schedule(static, 2)
    """
    def __new__(cls, *args, **options):
        if not(len(args) in [1, 2]):
            raise ValueError('Expecting 1 or 2 entries, '
                             'given {0}'.format(len(args)))

        kind = args[0]

        chunk_size = None
        if len(args) == 2:
            chunk_size = args[1]

        return Basic.__new__(cls, kind, chunk_size)

    @property
    def kind(self):
        return self._args[0]

    @property
    def chunk_size(self):
        return self._args[1]

    def _sympystr(self, printer):
        sstr = printer.doprint

        kind = sstr(self.kind)

        chunk_size = ''
        if self.chunk_size:
            chunk_size = ', {0}'.format(sstr(self.chunk_size))

        return 'schedule({0}{1})'.format(kind, chunk_size)

class OMP_OrderedClause(OMP):
    """
    OMP OrderedClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_OrderedClause
    >>> OMP_OrderedClause(2)
    ordered(2)
    >>> OMP_OrderedClause()
    ordered
    """
    def __new__(cls, *args, **options):
        if not(len(args) in [0, 1]):
            raise ValueError('Expecting 0 or 1 entries, '
                             'given {0}'.format(len(args)))

        n = None
        if len(args) == 1:
            n = args[0]

        return Basic.__new__(cls, n)

    @property
    def n_loops(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint

        n_loops = ''
        if self.n_loops:
            n_loops = '({0})'.format(sstr(self.n_loops))

        return 'ordered{0}'.format(n_loops)

class OMP_CollapseClause(OMP):
    """
    OMP CollapseClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_CollapseClause
    >>> OMP_CollapseClause(2)
    collapse(2)
    """
    def __new__(cls, *args, **options):
        if not(len(args) == 1):
            raise ValueError('Expecting 1 entry, '
                             'given {0}'.format(len(args)))

        n = args[0]
        return Basic.__new__(cls, n)

    @property
    def n_loops(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        n_loops = '{0}'.format(sstr(self.n_loops))

        return 'collapse({0})'.format(n_loops)

class OMP_LinearClause(OMP):
    """
    OMP LinearClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_LinearClause
    >>> OMP_LinearClause('x', 'y', 2)
    linear((x, y): 2)
    """
    # TODO check type of step => must be int, Integer
    def __new__(cls, *args, **options):
        variables = args[0:-1]
        step = args[-1]
        return Basic.__new__(cls, variables, step)

    @property
    def variables(self):
        return self._args[0]

    @property
    def step(self):
        return self._args[1]

    def _sympystr(self, printer):
        sstr = printer.doprint
        variables= ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        step = sstr(self.step)
        return "linear('{0}': {1})".format(variables, step)

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
    if isinstance(stmt, ForIterator):
        iterable = openmpfy(stmt.iterable, **options)
        target   = stmt.target
        body     = openmpfy(stmt.body, **options)

        cls_base = iterable.cls_base

        # ... if using OpenMP
        if ('openmp' in cls_base.options):
            # ...
            def _format_str(a):
                if isinstance(a, str):
                    return a.strip('\'')
                else:
                    return a
            # ...

            # ... get initial values for all attributs
            d_attributs = cls_base.attributs_as_dict

            d = {}
            for k,v in d_attributs.items():
                i = DottedName('self', k)
                d[k] = get_initial_value(cls_base, i)
            # ...

            # ... nowait
            nowait = d['_nowait']
            # ...

            # ... collapse
            collapse = None
            if not(d['_collapse'] is None):
                if not isinstance(d['_collapse'], Nil):
                    ls = [d['_collapse']]
                    collapse = OMP_CollapseClause(*ls)
            # ...

            # ... private
            private = None
            if not(d['_private'] is None):
                if not isinstance(d['_private'], Nil):
                    ls = d['_private']
                    ls = [_format_str(a) for a in ls]
                    private = OMP_PrivateClause(*ls)
            # ...

            # ... firstprivate
            firstprivate = None
            if not(d['_firstprivate'] is None):
                if not isinstance(d['_firstprivate'], Nil):
                    ls = d['_firstprivate']
                    ls = [_format_str(a) for a in ls]
                    firstprivate = OMP_FirstPrivateClause(*ls)
            # ...

            # ... lastprivate
            lastprivate = None
            if not(d['_lastprivate'] is None):
                if not isinstance(d['_lastprivate'], Nil):
                    ls = d['_lastprivate']
                    ls = [_format_str(a) for a in ls]
                    lastprivate = OMP_LastPrivateClause(*ls)
            # ...

            # ... reduction
            reduction = None
            if not(d['_reduction'] is None):
                if not isinstance(d['_reduction'], Nil):
                    ls = d['_reduction']
                    ls = [_format_str(a) for a in ls]
                    reduction = OMP_ReductionClause(*ls)
            # ...

            # ... schedule
            schedule = None
            if not(d['_schedule'] is None):
                if not isinstance(d['_schedule'], Nil):
                    ls = d['_schedule']
                    if isinstance(ls, str):
                        ls = [ls]

                    ls[0] = _format_str(ls[0])
                    schedule = OMP_ScheduleClause(*ls)
            # ...

            # ... ordered
            ordered = None
            if not(d['_ordered'] is None):
                if not isinstance(d['_ordered'], Nil):
                    ls = d['_ordered']

                    args = []
                    if isinstance(ls, (int, Integer)):
                        args.append(ls)

                    ordered = OMP_OrderedClause(*args)
            # ...

            # ... linear
            linear = None
            if not(d['_linear'] is None):
                if not isinstance(d['_linear'], Nil):
                    # we need to convert Tuple to list here
                    ls = list(d['_linear'])

                    if len(ls) < 2:
                        raise ValueError('Expecting at least 2 entries, '
                                         'given {0}'.format(len(ls)))

                    variables = [a.strip('\'') for a in ls[0:-1]]
                    ls[0:-1]  = variables

                    linear = OMP_LinearClause(*ls)
            # ...

            # ...
            clauses = (private, firstprivate, lastprivate,
                       reduction, schedule,
                       ordered, collapse, linear)
            clauses = [i for i in clauses if not(i is None)]
            clauses = Tuple(*clauses)
            # ...

            return OMP_For(target, iterable, body, clauses, nowait)
#            return ForIterator(target, iterable, body, strict=False)
        else:
            return ForIterator(target, iterable, body, strict=False)
        # ...
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
