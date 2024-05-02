# coding: utf-8

from sympy.core import Tuple
from sympy.utilities.iterables import iterable

from pyccel.ast.core import Module, Program
from pyccel.ast.core import Nil
from pyccel.ast.core import get_initial_value
from pyccel.ast.core import DottedName
from pyccel.ast.core import Variable
from pyccel.ast.core import ParallelBlock
from pyccel.ast.core import Tensor
from pyccel.ast.core import Import
from pyccel.ast.core import For, ForIterator, While, With, If
from pyccel.ast.core import FunctionDef, ClassDef
from pyccel.ast.core import ConstructorCall

from pyccel.ast.parallel.basic import Basic

__all__ = (
    'ACC',
    'ACC_Async',
    'ACC_Auto',
    'ACC_Bind',
    'ACC_Collapse',
    'ACC_Copy',
    'ACC_Copyin',
    'ACC_Copyout',
    'ACC_Create',
    'ACC_Default',
    'ACC_DefaultAsync',
    'ACC_Delete',
    'ACC_Device',
    'ACC_DeviceNum',
    'ACC_DevicePtr',
    'ACC_DeviceResident',
    'ACC_DeviceType',
    'ACC_Finalize',
    'ACC_FirstPrivate',
    'ACC_For',
    'ACC_Gang',
    'ACC_Host',
    'ACC_If',
    'ACC_IfPresent',
    'ACC_Independent',
    'ACC_Link',
    'ACC_NoHost',
    'ACC_NumGangs',
    'ACC_NumWorkers',
    'ACC_Parallel',
    'ACC_Present',
    'ACC_Private',
    'ACC_Reduction',
    'ACC_Self',
    'ACC_Seq',
    'ACC_Tile',
    'ACC_UseDevice',
    'ACC_Vector',
    'ACC_VectorLength',
    'ACC_Wait',
    'ACC_Worker',
    'accfy'
)

##########################################################
#               Base class for OpenACC
##########################################################
class ACC(Basic):
    """Base class for OpenACC."""
    pass

##########################################################
#                 Basic Statements
##########################################################
class ACC_Parallel(ParallelBlock, ACC):
    """
    ACC Parallel construct statement.

    Examples

    >>> from pyccel.parallel.openacc import ACC_Parallel
    >>> from pyccel.parallel.openacc import ACC_NumThread
    >>> from pyccel.parallel.openacc import ACC_Default
    >>> from pyccel.ast.core import Variable, Assign, Block
    >>> n = Variable('int', 'n')
    >>> x = Variable('int', 'x')
    >>> body = [Assign(x,2.*n + 1.), Assign(n, n + 1)]
    >>> variables = [x,n]
    >>> clauses = [ACC_NumThread(4), ACC_Default('shared')]
    >>> ACC_Parallel(clauses, variables, body)
    #pragma parallel num_threads(4) default(shared)
    x := 1.0 + 2.0*n
    n := 1 + n
    """
    _prefix = '#pragma'
    name = 'parallel'
    def __new__(cls, clauses, variables, body):
        if not iterable(clauses):
            raise TypeError('Expecting an iterable for clauses')

        _valid_clauses = (ACC_Async,
                          ACC_Wait,
                          ACC_NumGangs,
                          ACC_NumWorkers,
                          ACC_VectorLength,
                          ACC_DeviceType,
                          ACC_If,
                          ACC_Reduction,
                          ACC_Copy,
                          ACC_Copyin,
                          ACC_Copyout,
                          ACC_Create,
                          ACC_Present,
                          ACC_DevicePtr,
                          ACC_Private,
                          ACC_FirstPrivate,
                          ACC_Default)

        for clause in clauses:
            if not isinstance(clause, _valid_clauses):
                raise TypeError('Wrong clause for ACC_Parallel')

        return ParallelBlock.__new__(cls, clauses, variables, body)

class ACC_For(ForIterator, ACC):
    """
    ACC Loop construct statement.

    Examples

    """
    _prefix = '#pragma'
    name = 'do'
    def __new__(cls, loop, clauses):
        if not iterable(clauses):
            raise TypeError('Expecting an iterable for clauses')

        _valid_clauses = (ACC_Collapse,
                          ACC_Gang,
                          ACC_Worker,
                          ACC_Vector,
                          ACC_Seq,
                          ACC_Auto,
                          ACC_Tile,
                          ACC_DeviceType,
                          ACC_Independent,
                          ACC_Private,
                          ACC_Reduction)

        for clause in clauses:
            if not isinstance(clause, _valid_clauses):
                raise TypeError('Wrong clause for ACC_For, '
                               'given {0}'.format(type(clause)))

        return Basic.__new__(cls, loop, clauses)

    @property
    def loop(self):
        return self._args[0]

    @property
    def clauses(self):
        return self._args[1]

    @property
    def target(self):
        return self.loop.target

    @property
    def iterable(self):
        return self.loop.iterable

    @property
    def body(self):
        return self.loop.body
#################################################

#################################################
#                 Clauses
#################################################
class ACC_Async(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Async
    >>> ACC_Async('x', 'y')
    async(x, y)
    """
    name = 'async'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'async({})'.format(args)

class ACC_Auto(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Auto
    >>> ACC_Auto()
    auto
    """
    name = 'auto'

    def _sympystr(self, printer):
        return 'auto'

class ACC_Bind(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Bind
    >>> ACC_Bind('n')
    bind(n)
    """
    name = 'bind'
    def __new__(cls, *args, **options):
        if len(args) != 1:
            raise ValueError('Expecting 1 entry, '
                             'given {0}'.format(len(args)))

        variable = args[0]
        return Basic.__new__(cls, variable)

    @property
    def variable(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        variable = '{0}'.format(sstr(self.variable))

        return 'bind({0})'.format(variable)

class ACC_Collapse(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Collapse
    >>> ACC_Collapse(2)
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

class ACC_Copy(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Copy
    >>> ACC_Copy('x', 'y')
    copy(x, y)
    """
    name = 'copy'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'copy({})'.format(args)

class ACC_Copyin(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Copyin
    >>> ACC_Copyin('x', 'y')
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

class ACC_Copyout(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Copyout
    >>> ACC_Copyout('x', 'y')
    copyout(x, y)
    """
    name = 'copyout'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'copyout({})'.format(args)

class ACC_Create(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Create
    >>> ACC_Create('x', 'y')
    create(x, y)
    """
    name = 'create'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'create({})'.format(args)

class ACC_Default(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Default
    >>> ACC_Default('present')
    default(present)
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

class ACC_DefaultAsync(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_DefaultAsync
    >>> ACC_DefaultAsync('x', 'y')
    default_async(x, y)
    """
    name = 'default_async'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'default_async({})'.format(args)

class ACC_Delete(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Delete
    >>> ACC_Delete('x', 'y')
    delete(x, y)
    """
    name = 'delete'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'delete({})'.format(args)

class ACC_Device(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Device
    >>> ACC_Device('x', 'y')
    device(x, y)
    """
    name = 'device'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'device({})'.format(args)

class ACC_DeviceNum(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_DeviceNum
    >>> ACC_DeviceNum(2)
    device_num(2)
    """
    name = 'device_num'
    def __new__(cls, *args, **options):
        if len(args) != 1:
            raise ValueError('Expecting 1 entry, '
                             'given {0}'.format(len(args)))

        n = args[0]
        return Basic.__new__(cls, n)

    @property
    def n_device(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        n_device = '{0}'.format(sstr(self.n_device))

        return 'device_num({0})'.format(n_device)

class ACC_DevicePtr(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_DevicePtr
    >>> ACC_DevicePtr('x', 'y')
    deviceptr(x, y)
    """
    name = 'deviceptr'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'deviceptr({})'.format(args)

class ACC_DeviceResident(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_DeviceResident
    >>> ACC_DeviceResident('x', 'y')
    device_resident(x, y)
    """
    name = 'device_resident'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'device_resident({})'.format(args)

class ACC_DeviceType(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_DeviceType
    >>> ACC_DeviceType('x', 'y')
    device_type(x, y)
    """
    name = 'device_type'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'device_type({})'.format(args)

class ACC_Finalize(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Finalize
    >>> ACC_Finalize()
    finalize
    """
    name = 'finalize'

    def _sympystr(self, printer):
        return 'finalize'

class ACC_FirstPrivate(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_FirstPrivate
    >>> ACC_FirstPrivate('x', 'y')
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

class ACC_Gang(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Gang
    >>> ACC_Gang('x', 'y')
    gang(x, y)
    """
    name = 'gang'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'gang({})'.format(args)

class ACC_Host(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Host
    >>> ACC_Host('x', 'y')
    host(x, y)
    """
    name = 'host'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'host({})'.format(args)

class ACC_If(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_If
    >>> ACC_If(True)
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

class ACC_IfPresent(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_IfPresent
    >>> ACC_IfPresent()
    if_present
    """
    name = 'if_present'

    def _sympystr(self, printer):
        return 'if_present'

class ACC_Independent(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Independent
    >>> ACC_Independent()
    independent
    """
    name = 'independent'

    def _sympystr(self, printer):
        return 'independent'

class ACC_Link(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Link
    >>> ACC_Link('x', 'y')
    link(x, y)
    """
    name = 'link'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'link({})'.format(args)

class ACC_NoHost(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_NoHost
    >>> ACC_NoHost()
    nohost
    """
    name = 'nohost'

    def _sympystr(self, printer):
        return 'nohost'

class ACC_NumGangs(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_NumGangs
    >>> ACC_NumGangs(2)
    num_gangs(2)
    """
    name = 'num_gangs'
    def __new__(cls, *args, **options):
        if len(args) != 1:
            raise ValueError('Expecting 1 entry, '
                             'given {0}'.format(len(args)))

        n = args[0]
        return Basic.__new__(cls, n)

    @property
    def n_gang(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        n_gang = '{0}'.format(sstr(self.n_gang))

        return 'num_gangs({0})'.format(n_gang)

class ACC_NumWorkers(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_NumWorkers
    >>> ACC_NumWorkers(2)
    num_workers(2)
    """
    name = 'num_workers'
    def __new__(cls, *args, **options):
        if len(args) != 1:
            raise ValueError('Expecting 1 entry, '
                             'given {0}'.format(len(args)))

        n = args[0]
        return Basic.__new__(cls, n)

    @property
    def n_worker(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        n_worker = '{0}'.format(sstr(self.n_worker))

        return 'num_workers({0})'.format(n_worker)

class ACC_Present(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Present
    >>> ACC_Present('x', 'y')
    present(x, y)
    """
    name = 'present'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'present({})'.format(args)

class ACC_Private(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Private
    >>> ACC_Private('x', 'y')
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

class ACC_Reduction(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Reduction
    >>> ACC_Reduction('+', 'x', 'y')
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

class ACC_Self(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Self
    >>> ACC_Self('x', 'y')
    self(x, y)
    """
    name = 'self'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'self({})'.format(args)

class ACC_Seq(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Seq
    >>> ACC_Seq()
    seq
    """
    name = 'seq'

    def _sympystr(self, printer):
        return 'seq'

class ACC_Tile(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Tile
    >>> ACC_Tile('x', 'y')
    tile(x, y)
    """
    name = 'tile'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'tile({})'.format(args)

class ACC_UseDevice(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_UseDevice
    >>> ACC_UseDevice('x', 'y')
    use_device(x, y)
    """
    name = 'use_device'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'use_device({})'.format(args)

class ACC_Vector(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Vector
    >>> ACC_Vector('x', 'y')
    vector(x, y)
    """
    name = 'vector'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'vector({})'.format(args)

class ACC_VectorLength(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_VectorLength
    >>> ACC_VectorLength(2)
    vector_length(2)
    """
    name = 'vector_length'
    def __new__(cls, *args, **options):
        if len(args) != 1:
            raise ValueError('Expecting 1 entry, '
                             'given {0}'.format(len(args)))

        n = args[0]
        return Basic.__new__(cls, n)

    @property
    def n(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        return 'vector_length({0})'.format(sstr(self.n))

class ACC_Wait(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Wait
    >>> ACC_Wait('x', 'y')
    wait(x, y)
    """
    name = 'wait'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'wait({})'.format(args)

class ACC_Worker(ACC):
    """

    Examples

    >>> from pyccel.parallel.openacc import ACC_Worker
    >>> ACC_Worker('x', 'y')
    worker(x, y)
    """
    name = 'worker'
    def __new__(cls, *args, **options):
        return Basic.__new__(cls, args)

    @property
    def variables(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        args = ', '.join('{0}'.format(sstr(i)) for i in self.variables)
        return 'worker({})'.format(args)

##########################################################
#             useful functions
##########################################################
def accfy(stmt, **options):
    """
    Converts some statements to OpenACC statments.

    stmt: stmt, list
        statement or a list of statements
    """
    if isinstance(stmt, (list, tuple, Tuple)):
        return [accfy(i, **options) for i in stmt]

    if isinstance(stmt, Tensor):
        # TODO to implement
        return stmt

    if isinstance(stmt, ForIterator):
        iterable = accfy(stmt.iterable, **options)
        target   = stmt.target
        body     = accfy(stmt.body, **options)

        clauses = get_for_clauses(iterable)

        if (clauses is None):
            return ForIterator(target, iterable, body, strict=False)
        else:
            loop   = ForIterator(target, iterable, body, strict=False)
            return ACC_For(loop, clauses)

    if isinstance(stmt, For):
        iterable = accfy(stmt.iterable, **options)
        target   = stmt.target
        body     = accfy(stmt.body, **options)
        return For(target, iterable, body, strict=False)

    if isinstance(stmt, list):
        return [accfy(a, **options) for a in stmt]

    if isinstance(stmt, While):
        test = accfy(stmt.test, **options)
        body = accfy(stmt.body, **options)
        return While(test, body)

    if isinstance(stmt, With):
        test     = accfy(stmt.test, **options)
        body     = accfy(stmt.body, **options)
        settings = accfy(stmt.settings, **options)

        clauses = get_with_clauses(test)

        if (clauses is None):
            return With(test, body, settings)
        else:
            # TODO to be defined
            variables = []
            return ACC_Parallel(clauses, variables, body)

    if isinstance(stmt, If):
        args = []
        for block in stmt.args:
            test  = block[0]
            stmts = block[1]
            t = accfy(test,  **options)
            s = accfy(stmts, **options)
            args.append((t,s))
        return If(*args)

    if isinstance(stmt, FunctionDef):
        name        = accfy(stmt.name,        **options)
        arguments   = accfy(stmt.arguments,   **options)
        results     = accfy(stmt.results,     **options)
        body        = accfy(stmt.body,        **options)
        local_vars  = accfy(stmt.local_vars,  **options)
        global_vars = accfy(stmt.global_vars, **options)

        return FunctionDef(name, arguments, results, \
                           body, local_vars, global_vars)

    if isinstance(stmt, ClassDef):
        name        = accfy(stmt.name,        **options)
        attributs   = accfy(stmt.attributs,   **options)
        methods     = accfy(stmt.methods,     **options)
        options     = accfy(stmt.options,     **options)

        return ClassDef(name, attributs, methods, options)

    if isinstance(stmt, Module):
        name        = accfy(stmt.name,        **options)
        variables   = accfy(stmt.variables,   **options)
        funcs       = accfy(stmt.funcs    ,   **options)
        classes     = accfy(stmt.classes  ,   **options)
        imports     = accfy(stmt.imports  ,   **options)
        imports    += [Import('openacc')]

        return Module(name, variables, funcs, classes,
                      imports=imports)

    if isinstance(stmt, Program):
        name        = accfy(stmt.name,        **options)
        variables   = accfy(stmt.variables,   **options)
        funcs       = accfy(stmt.funcs    ,   **options)
        classes     = accfy(stmt.classes  ,   **options)
        imports     = accfy(stmt.imports  ,   **options)
        body        = accfy(stmt.body  ,   **options)
        modules     = accfy(stmt.modules  ,   **options)
        imports    += [Import('openacc')]

        return Program(name, variables, funcs, classes, body,
                       imports=imports, modules=modules)

    if isinstance(stmt, ParallelBlock):
        variables = stmt.variables
        body      = stmt.body
        clauses   = stmt.clauses

        return ACC_Parallel(clauses, variables, body)

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

        if not(('openacc' in cls_base.options) and ('with' in cls_base.options)):
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
    _async         = None
    _wait          = None
    _num_gangs     = None
    _num_workers   = None
    _vector_length = None
    _device_type   = None
    _if            = None
    _reduction     = None
    _copy          = None
    _copyin        = None
    _copyout       = None
    _create        = None
    _present       = None
    _deviceptr     = None
    _private       = None
    _firstprivate  = None
    _default       = None
    # ...

    # ... async
    if not(d['_async'] is None):
        if not isinstance(d['_async'], Nil):
            ls = d['_async']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _async = ACC_Async(*ls)
    # ...

    # ... copy
    if not(d['_copy'] is None):
        if not isinstance(d['_copy'], Nil):
            ls = d['_copy']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _copy = ACC_Copy(*ls)
    # ...

    # ... copyin
    if not(d['_copyin'] is None):
        if not isinstance(d['_copyin'], Nil):
            ls = d['_copyin']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _copyin = ACC_Copyin(*ls)
    # ...

    # ... copyout
    if not(d['_copyout'] is None):
        if not isinstance(d['_copyout'], Nil):
            ls = d['_copyout']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _copyout = ACC_Copyout(*ls)
    # ...

    # ... create
    if not(d['_create'] is None):
        if not isinstance(d['_create'], Nil):
            ls = d['_create']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _create = ACC_Copyin(*ls)
    # ...

    # ... default
    if not(d['_default'] is None):
        if not isinstance(d['_default'], Nil):
            ls = d['_default']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls[0] = _format_str(ls[0])
            _default = ACC_Default(*ls)
    # ...

    # ... deviceptr
    if not(d['_deviceptr'] is None):
        if not isinstance(d['_deviceptr'], Nil):
            ls = d['_deviceptr']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _deviceptr = ACC_DevicePtr(*ls)
    # ...

    # ... devicetype
    if not(d['_device_type'] is None):
        if not isinstance(d['_device_type'], Nil):
            ls = d['_device_type']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _device_type = ACC_DeviceType(*ls)
    # ...

    # ... firstprivate
    if not(d['_firstprivate'] is None):
        if not isinstance(d['_firstprivate'], Nil):
            ls = d['_firstprivate']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _firstprivate = ACC_FirstPrivate(*ls)
    # ...

    # ... if
    #     TODO improve this to take any boolean expression for arg.
    #     see OpenACC specifications
    if not(d['_if'] is None):
        if not isinstance(d['_if'], Nil):
            arg = d['_if']
            ls = [arg]
            _if = ACC_If(*ls)
    # ...

    # ... num_gangs
    #     TODO improve this to take any int expression for arg.
    #     see OpenACC specifications
    if not(d['_num_gangs'] is None):
        if not isinstance(d['_num_gangs'], Nil):
            arg = d['_num_gangs']
            ls = [arg]
            _num_gangs = ACC_NumGangs(*ls)
    # ...

    # ... num_workers
    #     TODO improve this to take any int expression for arg.
    #     see OpenACC specifications
    if not(d['_num_workers'] is None):
        if not isinstance(d['_num_workers'], Nil):
            arg = d['_num_workers']
            ls = [arg]
            _num_workers = ACC_NumWorkers(*ls)
    # ...

    # ... present
    if not(d['_present'] is None):
        if not isinstance(d['_present'], Nil):
            ls = d['_present']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _present = ACC_Present(*ls)
    # ...

    # ... private
    if not(d['_private'] is None):
        if not isinstance(d['_private'], Nil):
            ls = d['_private']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _private = ACC_Private(*ls)
    # ...

    # ... reduction
    if not(d['_reduction'] is None):
        if not isinstance(d['_reduction'], Nil):
            ls = d['_reduction']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _reduction = ACC_Reduction(*ls)
    # ...

    # ... vector_length
    if not(d['_vector_length'] is None):
        if not isinstance(d['_vector_length'], Nil):
            arg = d['_vector_length']
            ls = [arg]
            _vector_length = ACC_VectorLength(*ls)
    # ...

    # ... wait
    if not(d['_wait'] is None):
        if not isinstance(d['_wait'], Nil):
            ls = d['_wait']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _wait = ACC_Wait(*ls)
    # ...

    # ...
    clauses = (_async,
               _wait,
               _num_gangs,
               _num_workers,
               _vector_length,
               _device_type,
               _if,
               _reduction,
               _copy,
               _copyin,
               _copyout,
               _create,
               _present,
               _deviceptr,
               _private,
               _firstprivate,
               _default)

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

        if not(('openacc' in cls_base.options) and ('iterable' in cls_base.options)):
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
    _collapse    = None
    _gang        = None
    _worker      = None
    _vector      = None
    _seq         = None
    _auto        = None
    _tile        = None
    _device_type = None
    _independent = None
    _private     = None
    _reduction   = None
    # ...

    # ... auto
    if not(d['_auto'] is None):
        if not isinstance(d['_auto'], Nil):
            _auto = ACC_Auto()
    # ...

    # ... collapse
    if not(d['_collapse'] is None):
        if not isinstance(d['_collapse'], Nil):
            ls = [d['_collapse']]
            _collapse = ACC_Collapse(*ls)
    # ...

    # ... device_type
    if not(d['_device_type'] is None):
        if not isinstance(d['_device_type'], Nil):
            ls = d['_device_type']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _device_type = ACC_DeviceType(*ls)
    # ...

    # ... gang
    if not(d['_gang'] is None):
        if not isinstance(d['_gang'], Nil):
            ls = d['_gang']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _gang = ACC_Gang(*ls)
    # ...

    # ... independent
    if not(d['_independent'] is None):
        if not isinstance(d['_independent'], Nil):
            _independent = ACC_Independent()
    # ...

    # ... private
    if not(d['_private'] is None):
        if not isinstance(d['_private'], Nil):
            ls = d['_private']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _private = ACC_Private(*ls)
    # ...

    # ... reduction
    if not(d['_reduction'] is None):
        if not isinstance(d['_reduction'], Nil):
            ls = d['_reduction']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _reduction = ACC_Reduction(*ls)
    # ...

    # ... seq
    if not(d['_seq'] is None):
        if not isinstance(d['_seq'], Nil):
            _seq = ACC_Seq()
    # ...

    # ... tile
    if not(d['_tile'] is None):
        if not isinstance(d['_tile'], Nil):
            ls = d['_tile']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _tile = ACC_Tile(*ls)
    # ...

    # ... vector
    if not(d['_vector'] is None):
        if not isinstance(d['_vector'], Nil):
            ls = d['_vector']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _vector = ACC_Vector(*ls)
    # ...

    # ... worker
    if not(d['_worker'] is None):
        if not isinstance(d['_worker'], Nil):
            ls = d['_worker']
            if not isinstance(ls, (list, tuple, Tuple)):
                ls = [ls]

            ls = [_format_str(a) for a in ls]
            _worker = ACC_Worker(*ls)
    # ...

    # ...
    clauses = (_collapse,
               _gang,
               _worker,
               _vector,
               _seq,
               _auto,
               _tile,
               _device_type,
               _independent,
               _private,
               _reduction)

    clauses = [i for i in clauses if not(i is None)]
    clauses = Tuple(*clauses)
    # ...

    return clauses
# ...
