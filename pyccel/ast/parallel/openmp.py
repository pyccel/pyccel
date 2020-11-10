# coding: utf-8

from sympy.core import Tuple
from sympy.utilities.iterables import iterable
from sympy import Integer as sp_Integer

from pyccel.ast.core import Module, Program
from pyccel.ast.core import Nil
from pyccel.ast.core import get_initial_value
from pyccel.ast.core import DottedName
from pyccel.ast.core import Variable
from pyccel.ast.core import ParallelBlock
from pyccel.ast.core import Import
from pyccel.ast.core import For, ForIterator, While, With, If
from pyccel.ast.core import FunctionDef, ClassDef
from pyccel.ast.core import ConstructorCall
from pyccel.ast.core import Tensor

from pyccel.ast.parallel.basic import Basic

__all__ = (
    'OMP',
    'OMP_Collapse',
    'OMP_Copyin',
    'OMP_Default',
    'OMP_FirstPrivate',
    'OMP_For',
    'OMP_If',
    'OMP_LastPrivate',
    'OMP_Linear',
    'OMP_NumThread',
    'OMP_Ordered',
    'OMP_Parallel',
    'OMP_Private',
    'OMP_ProcBind',
    'OMP_Reduction',
    'OMP_Schedule',
    'OMP_Shared',
    'ompfy'
)

##########################################################
#               Base class for OpenMP
##########################################################
class OMP(Basic):
    """Base class for OpenMP."""
    pass

##########################################################
#                 Basic Statements
##########################################################
class OMP_Parallel(ParallelBlock, OMP):
    """
    OMP Parallel construct statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_Parallel
    >>> from pyccel.parallel.openmp import OMP_NumThread
    >>> from pyccel.parallel.openmp import OMP_Default
    >>> from pyccel.ast.core import Variable, Assign, Block
    >>> n = Variable('int', 'n')
    >>> x = Variable('int', 'x')
    >>> body = [Assign(x,2.*n + 1.), Assign(n, n + 1)]
    >>> variables = [x,n]
    >>> clauses = [OMP_NumThread(4), OMP_Default('shared')]
    >>> OMP_Parallel(clauses, variables, body)
    #pragma parallel num_threads(4) default(shared)
    x := 1.0 + 2.0*n
    n := 1 + n
    """
    _prefix = '#pragma'
    name = 'parallel'
    def __new__(cls, clauses, variables, body):
        if not iterable(clauses):
            raise TypeError('Expecting an iterable for clauses')

        _valid_clauses = (OMP_NumThread, \
                          OMP_If, \
                          OMP_Default, \
                          OMP_Private, \
                          OMP_Shared, \
                          OMP_FirstPrivate, \
                          OMP_Copyin, \
                          OMP_Reduction, \
                          OMP_ProcBind)

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
    name = 'do'
    def __new__(cls, loop, clauses, nowait):
        if not iterable(clauses):
            raise TypeError('Expecting an iterable for clauses')

        _valid_clauses = (OMP_Schedule, \
                          OMP_Private, \
                          OMP_FirstPrivate, \
                          OMP_LastPrivate, \
                          OMP_Reduction, \
                          OMP_Collapse, \
                          OMP_Ordered, \
                          OMP_Linear)

        for clause in clauses:
            if not isinstance(clause, _valid_clauses):
                raise TypeError('Wrong clause for OMP_For, '
                               'given {0}'.format(type(clause)))

        return Basic.__new__(cls, loop, clauses, nowait)

    @property
    def loop(self):
        return self._args[0]

    @property
    def clauses(self):
        return self._args[1]

    @property
    def nowait(self):
        return self._args[2]

    @property
    def target(self):
        return self.loop.target

    @property
    def iterable(self):
        return self.loop.iterable

    @property
    def body(self):
        return self.loop.body


class OMP_NumThread(OMP):
    """
    OMP ParallelNumThreadClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_NumThread
    >>> OMP_NumThread(4)
    num_threads(4)
    """
    name = 'num_threads'
    def __new__(cls, *args, **options):
        num_threads = args[0]
        return Basic.__new__(cls, num_threads)

    @property
    def num_threads(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        return 'num_threads({})'.format(sstr(self.num_threads))

class OMP_If(OMP):
    """
    OMP ParallelIfClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_If
    >>> OMP_If(True)
    if (True)
    """
    name = 'if'
    def __new__(cls, *args, **options):
        test = args[0]
        return Basic.__new__(cls, test)

    @property
    def test(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        return 'if({})'.format(sstr(self.test))

class OMP_Default(OMP):
    """
    OMP ParallelDefaultClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_Default
    >>> OMP_Default('shared')
    default(shared)
    """
    name = None
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

class OMP_ProcBind(OMP):
    """
    OMP ParallelProcBindClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_ProcBind
    >>> OMP_ProcBind('master')
    proc_bind(master)
    """
    name = 'proc_bind'
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

class OMP_Private(OMP):
    """
    OMP PrivateClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_Private
    >>> OMP_Private('x', 'y')
    private(x, y)
    """
    name = 'private'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'private({})'.format(args)

class OMP_Shared(OMP):
    """
    OMP SharedClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_Shared
    >>> OMP_Shared('x', 'y')
    shared(x, y)
    """
    name = 'shared'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'shared({})'.format(args)

class OMP_FirstPrivate(OMP):
    """
    OMP FirstPrivateClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_FirstPrivate
    >>> OMP_FirstPrivate('x', 'y')
    firstprivate(x, y)
    """
    name = 'firstprivate'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'firstprivate({})'.format(args)

class OMP_LastPrivate(OMP):
    """
    OMP LastPrivateClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_LastPrivate
    >>> OMP_LastPrivate('x', 'y')
    lastprivate(x, y)
    """
    name = 'lastprivate'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'lastprivate({})'.format(args)

class OMP_Copyin(OMP):
    """
    OMP CopyinClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_Copyin
    >>> OMP_Copyin('x', 'y')
    copyin(x, y)
    """
    name = 'copyin'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'copyin({})'.format(args)

class OMP_Reduction(OMP):
    """
    OMP ReductionClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_Reduction
    >>> OMP_Reduction('+', 'x', 'y')
    reduction('+': (x, y))
    """
    name = 'reduction'
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
        return "reduction({0}: {1})".format(op, args)

class OMP_Schedule(OMP):
    """
    OMP ScheduleClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_Schedule
    >>> OMP_Schedule('static', 2)
    schedule(static, 2)
    """
    name = 'schedule'
    def __new__(cls, *args, **options):
        if len(args) not in [1, 2]:
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

class OMP_Ordered(OMP):
    """
    OMP OrderedClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_Ordered
    >>> OMP_Ordered(2)
    ordered(2)
    >>> OMP_Ordered()
    ordered
    """
    name = 'ordered'
    def __new__(cls, *args, **options):
        if len(args) not in [0, 1]:
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

class OMP_Collapse(OMP):
    """
    OMP CollapseClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_Collapse
    >>> OMP_Collapse(2)
    collapse(2)
    """
    name = 'collapse'
    def __new__(cls, *args, **options):
        if len(args) != 1:
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

class OMP_Linear(OMP):
    """
    OMP LinearClause statement.

    Examples

    >>> from pyccel.parallel.openmp import OMP_Linear
    >>> OMP_Linear('x', 'y', 2)
    linear((x, y): 2)
    """
    name = 'linear'
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
        return "linear({0}: {1})".format(variables, step)

##########################################################


##########################################################
#             useful functions
##########################################################
def ompfy(stmt, **options):
    """
    Converts some statements to OpenMP statments.

    stmt: stmt, list
        statement or a list of statements
    """
    if isinstance(stmt, (list, tuple, Tuple)):
        return [ompfy(i, **options) for i in stmt]

    if isinstance(stmt, Tensor):
        # TODO to implement
        return stmt

    if isinstance(stmt, ForIterator):
        iterable = ompfy(stmt.iterable, **options)
        target   = stmt.target
        body     = ompfy(stmt.body, **options)

        info, clauses = get_for_clauses(iterable)

        if (clauses is None):
            return ForIterator(target, iterable, body, strict=False)
        else:
            loop   = ForIterator(target, iterable, body, strict=False)
            nowait = info['nowait']
            return OMP_For(loop, clauses, nowait)

    if isinstance(stmt, For):
        iterable = ompfy(stmt.iterable, **options)
        target   = stmt.target
        body     = ompfy(stmt.body, **options)
        return For(target, iterable, body, strict=False)

    if isinstance(stmt, list):
        return [ompfy(a, **options) for a in stmt]

    if isinstance(stmt, While):
        test = ompfy(stmt.test, **options)
        body = ompfy(stmt.body, **options)
        return While(test, body)

    if isinstance(stmt, With):
        test     = ompfy(stmt.test, **options)
        body     = ompfy(stmt.body, **options)
        settings = ompfy(stmt.settings, **options)

        clauses = get_with_clauses(test)

        if (clauses is None):
            return With(test, body, settings)
        else:
            # TODO to be defined
            variables = []
            return OMP_Parallel(clauses, variables, body)

    if isinstance(stmt, If):
        args = []
        for block in stmt.args:
            test  = block[0]
            stmts = block[1]
            t = ompfy(test,  **options)
            s = ompfy(stmts, **options)
            args.append((t,s))
        return If(*args)

    if isinstance(stmt, FunctionDef):
        name        = ompfy(stmt.name,        **options)
        arguments   = ompfy(stmt.arguments,   **options)
        results     = ompfy(stmt.results,     **options)
        body        = ompfy(stmt.body,        **options)
        local_vars  = ompfy(stmt.local_vars,  **options)
        global_vars = ompfy(stmt.global_vars, **options)

        return FunctionDef(name, arguments, results,
                           body, local_vars, global_vars)

    if isinstance(stmt, ClassDef):
        name        = ompfy(stmt.name,        **options)
        attributs   = ompfy(stmt.attributs,   **options)
        methods     = ompfy(stmt.methods,     **options)
        options     = ompfy(stmt.options,     **options)

        return ClassDef(name, attributs, methods, options)

    if isinstance(stmt, Module):
        name        = ompfy(stmt.name,        **options)
        variables   = ompfy(stmt.variables,   **options)
        funcs       = ompfy(stmt.funcs    ,   **options)
        classes     = ompfy(stmt.classes  ,   **options)
        imports     = ompfy(stmt.imports  ,   **options)
        imports    += [Import('omp_lib')]

        return Module(name, variables, funcs, classes,
                      imports=imports)

    if isinstance(stmt, Program):
        name        = ompfy(stmt.name,        **options)
        variables   = ompfy(stmt.variables,   **options)
        funcs       = ompfy(stmt.funcs    ,   **options)
        classes     = ompfy(stmt.classes  ,   **options)
        imports     = ompfy(stmt.imports  ,   **options)
        body        = ompfy(stmt.body  ,   **options)
        modules     = ompfy(stmt.modules  ,   **options)
        imports    += [Import('omp_lib')]

        return Program(name, variables, funcs, classes, body,
                       imports=imports, modules=modules)

    if isinstance(stmt, ParallelBlock):
        variables = stmt.variables
        body      = stmt.body
        clauses   = stmt.clauses

        return OMP_Parallel(clauses, variables, body)

    return stmt
##########################################################

# ...
def get_with_clauses(expr):
    # ...
    def _format_str(a):
        if isinstance(a, str):
            return a.strip('\'')
        else:
            return a
    # ...

    # ...
    d_attributs = {}
    d_args      = {}
    # ...

    # ... we first create a dictionary of attributs
    if isinstance(expr, Variable):
        if expr.cls_base:
            d_attributs = expr.cls_base.attributs_as_dict

    elif isinstance(expr, ConstructorCall):
        attrs = expr.attributs
        for i in attrs:
            d_attributs[str(i).replace('self.', '')] = i
    # ...

    # ...
    if not d_attributs:
        raise ValueError('Can not find attributs')
    # ...

    # ...
    if isinstance(expr, Variable):
        cls_base = expr.cls_base

        if not cls_base:
            return None

        if not(('openmp' in cls_base.options) and ('with' in cls_base.options)):
            return None
    elif isinstance(expr, ConstructorCall):
        # arguments[0] is 'self'
        # TODO must be improved in syntax, so that a['value'] is a sympy object
        for a in expr.arguments[1:]:
            if isinstance(a, dict):
                # we add '_' tp be conform with the private variables convention
                d_args['_{0}'.format(a['key'])] = a['value']
    else:
        return None
    # ...

    # ... get initial values for all attributs
    #     TODO do we keep 'self' hard coded?
    d = {}
    for k in d_attributs.keys():
        i = DottedName('self', k)
        d[k] = get_initial_value(expr, i)
    # ...

    # ... update the dictionary with the class parameters
    for k,v in d_args.items():
        d[k] = d_args[k]
    # ...

    # ... initial values for clauses
    private      = None
    firstprivate = None
    shared       = None
    reduction    = None
    copyin       = None
    default      = None
    proc_bind    = None
    num_threads  = None
    if_test      = None
    # ...

    # ... private
    if not(d['_private'] is None):
        if not isinstance(d['_private'], Nil):
            ls = d['_private']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            private = OMP_Private(*ls)
    # ...

    # ... firstprivate
    if not(d['_firstprivate'] is None):
        if not isinstance(d['_firstprivate'], Nil):
            ls = d['_firstprivate']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            firstprivate = OMP_FirstPrivate(*ls)
    # ...

    # ... shared
    if not(d['_shared'] is None):
        if not isinstance(d['_shared'], Nil):
            ls = d['_shared']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            shared = OMP_Shared(*ls)
    # ...

    # ... reduction
    if not(d['_reduction'] is None):
        if not isinstance(d['_reduction'], Nil):
            ls = d['_reduction']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            reduction = OMP_Reduction(*ls)
    # ...

    # ... copyin
    if not(d['_copyin'] is None):
        if not isinstance(d['_copyin'], Nil):
            ls = d['_copyin']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            copyin = OMP_Copyin(*ls)
    # ...

    # ... default
    if not(d['_default'] is None):
        if not isinstance(d['_default'], Nil):
            ls = d['_default']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls[0] = _format_str(ls[0])
            default = OMP_Default(*ls)
    # ...

    # ... proc_bind
    if not(d['_proc_bind'] is None):
        if not isinstance(d['_proc_bind'], Nil):
            ls = d['_proc_bind']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls[0] = _format_str(ls[0])
            proc_bind = OMP_ProcBind(*ls)
    # ...

    # ... num_threads
    #     TODO improve this to take any int expression for arg.
    #     see OpenMP specifications for num_threads clause
    if not(d['_num_threads'] is None):
        if not isinstance(d['_num_threads'], Nil):
            arg = d['_num_threads']
            ls = [arg]
            num_threads = OMP_NumThread(*ls)
    # ...

    # ... if_test
    #     TODO improve this to take any boolean expression for arg.
    #     see OpenMP specifications for if_test clause
    if not(d['_if_test'] is None):
        if not isinstance(d['_if_test'], Nil):
            arg = d['_if_test']
            ls = [arg]
            if_test = OMP_If(*ls)
    # ...

    # ...
    clauses = (private, firstprivate, shared,
               reduction, default, copyin,
               proc_bind, num_threads, if_test)
    clauses = [i for i in clauses if not(i is None)]
    clauses = Tuple(*clauses)
    # ...

    return clauses
# ...

# ...
def get_for_clauses(expr):
    # ...
    def _format_str(a):
        if isinstance(a, str):
            return a.strip('\'')
        else:
            return a
    # ...

    # ...
    d_attributs = {}
    d_args      = {}
    # ...

    # ... we first create a dictionary of attributs
    if isinstance(expr, Variable):
        if expr.cls_base:
            d_attributs = expr.cls_base.attributs_as_dict
    elif isinstance(expr, ConstructorCall):
        attrs = expr.attributs
        for i in attrs:
            d_attributs[str(i).replace('self.', '')] = i
    # ...

    # ...
    if not d_attributs:
        raise ValueError('Can not find attributs')
    # ...

    # ...
    if isinstance(expr, Variable):
        cls_base = expr.cls_base

        if not cls_base:
            return None, None

        if not(('openmp' in cls_base.options) and ('iterable' in cls_base.options)):
            return None, None
    elif isinstance(expr, ConstructorCall):
        # arguments[0] is 'self'
        # TODO must be improved in syntax, so that a['value'] is a sympy object
        for a in expr.arguments[1:]:
            if isinstance(a, dict):
                # we add '_' tp be conform with the private variables convention
                d_args['_{0}'.format(a['key'])] = a['value']
    else:
        return None, None
    # ...

    # ... get initial values for all attributs
    #     TODO do we keep 'self' hard coded?
    d = {}
    for k in d_attributs.keys():
        i = DottedName('self', k)
        d[k] = get_initial_value(expr, i)
    # ...

    # ... update the dictionary with the class parameters
    for k,v in d_args.items():
        d[k] = d_args[k]
    # ...

    # ... initial values for clauses
    nowait       = None

    collapse     = None
    private      = None
    firstprivate = None
    lastprivate  = None
    reduction    = None
    schedule     = None
    ordered      = None
    linear       = None
    # ...

    # ... nowait
    nowait = d['_nowait']
    # ...

    # ... collapse
    if not(d['_collapse'] is None):
        if not isinstance(d['_collapse'], Nil):
            ls = [d['_collapse']]
            collapse = OMP_Collapse(*ls)
    # ...

    # ... private
    if not(d['_private'] is None):
        if not isinstance(d['_private'], Nil):
            ls = d['_private']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            private = OMP_Private(*ls)
    # ...

    # ... firstprivate
    if not(d['_firstprivate'] is None):
        if not isinstance(d['_firstprivate'], Nil):
            ls = d['_firstprivate']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            firstprivate = OMP_FirstPrivate(*ls)
    # ...

    # ... lastprivate
    if not(d['_lastprivate'] is None):
        if not isinstance(d['_lastprivate'], Nil):
            ls = d['_lastprivate']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            lastprivate = OMP_LastPrivate(*ls)
    # ...

    # ... reduction
    if not(d['_reduction'] is None):
        if not isinstance(d['_reduction'], Nil):
            ls = d['_reduction']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            reduction = OMP_Reduction(*ls)
    # ...

    # ... schedule
    if not(d['_schedule'] is None):
        if not isinstance(d['_schedule'], Nil):
            ls = d['_schedule']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls[0] = _format_str(ls[0])
            schedule = OMP_Schedule(*ls)
    # ...

    # ... ordered
    if not(d['_ordered'] is None):
        if not isinstance(d['_ordered'], Nil):
            ls = d['_ordered']

            args = []
            if isinstance(ls, (int, sp_Integer)):
                args.append(ls)

            ordered = OMP_Ordered(*args)
    # ...

    # ... linear
    if not(d['_linear'] is None):
        if not isinstance(d['_linear'], Nil):
            # we need to convert Tuple to list here
            ls = list(d['_linear'])

            if len(ls) < 2:
                raise ValueError('Expecting at least 2 entries, '
                                 'given {0}'.format(len(ls)))

            variables = [a.strip('\'') for a in ls[0:-1]]
            ls[0:-1]  = variables

            linear = OMP_Linear(*ls)
    # ...

    # ...
    clauses = (private, firstprivate, lastprivate,
               reduction, schedule,
               ordered, collapse, linear)
    clauses = [i for i in clauses if not(i is None)]
    clauses = Tuple(*clauses)
    # ...

    # ...
    info = {}
    info['nowait'] = nowait
    # ...

    return info, clauses
# ...
