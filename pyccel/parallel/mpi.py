# coding: utf-8

# TODO - MPI_comm_gatherv: needs a new data structure for the variable

from itertools import groupby
import numpy as np

from sympy.core.symbol  import Symbol
from sympy.core.numbers import Integer
from sympy.core.compatibility import with_metaclass
from sympy.core.singleton import Singleton
from sympy.logic.boolalg import Boolean, BooleanTrue, BooleanFalse
from sympy.core import Tuple
from sympy.utilities.iterables import iterable

from pyccel.types.ast import Variable, IndexedVariable, IndexedElement
from pyccel.types.ast import Assign, Declare
from pyccel.types.ast import NativeBool, NativeFloat
from pyccel.types.ast import NativeComplex, NativeDouble, NativeInteger
from pyccel.types.ast import DataType
from pyccel.types.ast import DataTypeFactory
from pyccel.types.ast import Block
from pyccel.types.ast import Range, Tensor
from pyccel.types.ast import Zeros
from pyccel.types.ast import Ones

from pyccel.types.ast import For, While, If, Del, Sync
from pyccel.types.ast import FunctionDef, ClassDef

from pyccel.parallel.basic        import Basic
from pyccel.parallel.communicator import UniversalCommunicator


def get_shape(expr):
    """Returns the shape of a given variable."""
    if not isinstance(expr, (Variable, IndexedVariable, IndexedElement)):
        txt  = 'shape is only defined for Variable, IndexedVariable, IndexedElement.'
        txt += 'given {0}'.format(type(expr))
        raise TypeError(txt)

    if isinstance(expr, (Variable, IndexedVariable)):
        shape = expr.shape
        if shape is None:
            return 1
        elif isinstance(shape, (list, tuple, Tuple)):
            n = 1
            for i in shape:
                n *= i
            return n
        else:
            return shape
    elif isinstance(expr, IndexedElement):
        return get_shape(expr.base)

##########################################################
#               Base class for MPI
##########################################################
class MPI(Basic):
    """Base class for MPI."""
    pass
##########################################################

##########################################################
#                 Basic Statements
##########################################################
class MPI_Assign(Assign, MPI):
    """MPI statement that can be written as an assignment in pyccel."""
    pass

class MPI_Declare(Declare, MPI):
    """MPI declaration of a variable."""
    pass
##########################################################

##########################################################
#                  Constants
##########################################################
class MPI_status_size(MPI):
    """
    Represents the status size in mpi.

    Examples

    >>> from pyccel.parallel.mpi import MPI_status_size
    >>> MPI_status_size()
    mpi_status_size
    """
    is_integer     = True

    def _sympystr(self, printer):
        sstr = printer.doprint
        return 'mpi_status_size'

class MPI_proc_null(MPI):
    """
    Represents the null process in mpi.

    Examples

    >>> from pyccel.parallel.mpi import MPI_proc_null
    >>> MPI_proc_null()
    mpi_proc_null
    """
    is_integer     = True

    def _sympystr(self, printer):
        sstr = printer.doprint
        return 'mpi_proc_null'

class MPI_comm(MPI):
    """
    Represents a communicator in mpi.

    Examples

    >>> from pyccel.parallel.mpi import MPI_comm
    >>> MPI_comm('comm')
    comm
    """
    is_integer = True

    def __new__(cls, *args, **options):
        if len(args) == 1:
            name = args[0]
            if not isinstance(name, str):
                raise TypeError('Expecting a string')

        return super(MPI_comm, cls).__new__(cls, *args, **options)

    @property
    def name(self):
        return self._args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        return sstr(self.name)

class MPI_comm_world(UniversalCommunicator, MPI_comm):
    """
    Represents the world comm in mpi.

    Examples

    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> MPI_comm_world()
    mpi_comm_world
    """
    is_integer     = True

    def _sympystr(self, printer):
        sstr = printer.doprint
        return 'mpi_comm_world'
##########################################################

##########################################################
#                      Datatypes
##########################################################
# TODO to be removed
class MPI_status_type(DataType):
    """Represents the datatype of MPI status."""
    pass

class MPI_INTEGER(DataType):
    _name = 'MPI_INTEGER'

    def _sympystr(self, printer):
        return 'MPI_INTEGER'

class MPI_REAL(DataType):
    _name = 'MPI_REAL'

    def _sympystr(self, printer):
        return 'MPI_REAL'

class MPI_DOUBLE(DataType):
    _name = 'MPI_DOUBLE'

    def _sympystr(self, printer):
        return 'MPI_DOUBLE'

class MPI_COMPLEX(DataType):
    _name = 'MPI_COMPLEX'

    def _sympystr(self, printer):
        return 'MPI_COMPLEX'

class MPI_LOGICAL(DataType):
    _name = 'MPI_LOGICAL'

    def _sympystr(self, printer):
        return 'MPI_LOGICAL'

class MPI_CHARACTER(DataType):
    _name = 'MPI_CHARACTER'

    def _sympystr(self, printer):
        return 'MPI_CHARACTER'

def mpi_datatype(dtype):
    """Converts Pyccel datatypes into MPI datatypes."""
    if isinstance(dtype, NativeInteger):
        return 'MPI_INT'
    elif isinstance(dtype, NativeFloat):
        return 'MPI_REAL'
    elif isinstance(dtype, NativeDouble):
        return 'MPI_DOUBLE'
    elif isinstance(dtype, NativeBool):
        return 'MPI_LOGICAL'
    elif isinstance(dtype, NativeComplex):
        return 'MPI_COMPLEX'
    else:
        raise TypeError("Uncovered datatype : ", type(dtype))
##########################################################

##########################################################
#                    Operations
##########################################################
# The following are defined to be sympy approved nodes. If there is something
# smaller that could be used, that would be preferable. We only use them as
# tokens.

class MPI_Operation(with_metaclass(Singleton, Basic)):
    """Base type for native operands."""
    pass

class MPI_SUM(MPI_Operation):
    _name   = 'MPI_SUM'
    _symbol = '+'

    def _sympystr(self, printer):
        return 'MPI_SUM'

class MPI_PROD(MPI_Operation):
    _name   = 'MPI_PROD'
    _symbol = '*'

    def _sympystr(self, printer):
        return 'MPI_PROD'

_op_registry = {'+': MPI_SUM(), '*': MPI_PROD()}


def mpi_operation(op):
    """Returns the operator singleton for the given operator"""

    if op.lower() not in _op_registry:
        raise ValueError("Unrecognized MPI operation " + op)
    return _op_registry[op]
##########################################################

##########################################################
#           Communicator Accessors
##########################################################
class MPI_comm_size(MPI):
    """
    Represents the size of a given communicator.

    Examples

    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_size
    >>> comm = MPI_comm_world()
    >>> MPI_comm_size(comm)
    mpi_comm_world.size
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_size, cls).__new__(cls, *args, **options)

    @property
    def comm(self):
        return self.args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0}.{1}'.format(sstr(self.comm), 'size')

class MPI_comm_rank(MPI):
    """
    Represents the process rank within a given communicator.

    Examples

    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_rank
    >>> comm = MPI_comm_world()
    >>> MPI_comm_rank(comm)
    mpi_comm_world.rank
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_rank, cls).__new__(cls, *args, **options)

    @property
    def comm(self):
        return self.args[0]
##########################################################

##########################################################
#          Point-to-Point Communication
##########################################################
class MPI_comm_recv(MPI):
    """
    Represents the MPI_recv statement.
    MPI_recv syntax is
    `MPI_RECV (data, count, datatype, source, tag, comm, status)`

    data:
        initial address of receive buffer (choice) [OUT]

    count:
        number of elements in receive buffer (non-negative integer) [IN]

    datatype:
        datatype of each receive buffer element (handle) [IN]

    source:
        rank of source or MPI_ANY_SOURCE (integer) [IN]

    tag:
        message tag or MPI_ANY_TAG (integer) [IN]

    comm:
        communicator (handle) [IN]

    status:
        status object (Status) [OUT]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_recv
    >>> n = Variable('int', 'n')
    >>> x = Variable('double', 'x', rank=2, shape=(n,2), allocatable=True)
    >>> source = Variable('int', 'source')
    >>> tag  = Variable('int', 'tag')
    >>> comm = MPI_comm_world()
    >>> MPI_comm_recv(x, source, tag, comm)
    MPI_recv (x, 2*n, MPI_DOUBLE, source, tag, mpi_comm_world, i_mpi_status, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_recv, cls).__new__(cls, *args, **options)

    @property
    def data(self):
        return self.args[0]

    @property
    def source(self):
        return self.args[1]

    @property
    def tag(self):
        return self.args[2]

    @property
    def comm(self):
        return self.args[3]

    @property
    def count(self):
        return get_shape(self.data)

    @property
    def datatype(self):
        return mpi_datatype(self.data.dtype)

    def _sympystr(self, printer):
        sstr = printer.doprint

        data    = self.data
        count   = self.count
        dtype   = self.datatype
        source  = self.source
        tag     = self.tag
        comm    = self.comm
        ierr    = MPI_ERROR
        istatus = MPI_STATUS

        args = (data, count, dtype, source, tag, comm, istatus, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_recv ({0})'.format(args)

        return code

class MPI_comm_send(MPI):
    """
    Represents the MPI_send statement.
    MPI_send syntax is
    `MPI_SEND (data, count, datatype, dest, tag, comm)`

    data:
        initial address of send buffer (choice) [IN]
    count:
        number of elements in send buffer (non-negative integer) [IN]
    datatype:
        datatype of each send buffer element (handle) [IN]
    dest:
        rank of destination (integer) [IN]
    tag:
        message tag (integer) [IN]
    comm:
        communicator (handle) [IN]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_send
    >>> n = Variable('int', 'n')
    >>> x = Variable('double', 'x', rank=2, shape=(n,2), allocatable=True)
    >>> dest = Variable('int', 'dest')
    >>> tag  = Variable('int', 'tag')
    >>> comm = MPI_comm_world()
    >>> MPI_comm_send(x, dest, tag, comm)
    MPI_send (x, 2*n, MPI_DOUBLE, dest, tag, mpi_comm_world, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_send, cls).__new__(cls, *args, **options)

    @property
    def data(self):
        return self.args[0]

    @property
    def dest(self):
        return self.args[1]

    @property
    def tag(self):
        return self.args[2]

    @property
    def comm(self):
        return self.args[3]

    @property
    def count(self):
        return get_shape(self.data)

    @property
    def datatype(self):
        return mpi_datatype(self.data.dtype)

    def _sympystr(self, printer):
        sstr = printer.doprint

        data  = self.data
        count = self.count
        dtype = self.datatype
        dest  = self.dest
        tag   = self.tag
        comm  = self.comm
        ierr  = MPI_ERROR

        args = (data, count, dtype, dest, tag, comm, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_send ({0})'.format(args)
        return code

##########################################################

##########################################################
#
##########################################################
class MPI_comm_irecv(MPI):
    """
    Represents the MPI_irecv statement.
    MPI_irecv syntax is
    `MPI_IRECV (data, count, datatype, source, tag, comm, status)`

    data:
        initial address of receive buffer (choice) [OUT]
    count:
        number of elements in receive buffer (non-negative integer) [IN]
    datatype:
        datatype of each receive buffer element (handle) [IN]
    source:
        rank of source or MPI_ANY_SOURCE (integer) [IN]
    tag:
        message tag or MPI_ANY_TAG (integer) [IN]
    comm:
        communicator (handle) [IN]
    status:
        status object (Status) [OUT]
    request:
        communication request [OUT]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_irecv
    >>> n = Variable('int', 'n')
    >>> x = Variable('double', 'x', rank=2, shape=(n,2), allocatable=True)
    >>> source = Variable('int', 'source')
    >>> tag  = Variable('int', 'tag')
    >>> requests = Variable('int', 'requests', rank=1, shape=4, allocatable=True)
    >>> comm = MPI_comm_world()
    >>> MPI_comm_irecv(x, source, tag, requests, comm)
    MPI_irecv (x, 2*n, MPI_DOUBLE, source, tag, mpi_comm_world, requests, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_irecv, cls).__new__(cls, *args, **options)

    @property
    def data(self):
        return self.args[0]

    @property
    def source(self):
        return self.args[1]

    @property
    def tag(self):
        return self.args[2]

    @property
    def request(self):
        return self.args[3]

    @property
    def comm(self):
        return self.args[4]

    @property
    def count(self):
        return get_shape(self.data)

    @property
    def datatype(self):
        return mpi_datatype(self.data.dtype)

    def _sympystr(self, printer):
        sstr = printer.doprint

        data    = self.data
        count   = self.count
        dtype   = self.datatype
        source  = self.source
        tag     = self.tag
        comm    = self.comm
        request = self.request
        ierr    = MPI_ERROR

        args = (data, count, dtype, source, tag, comm, request, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_irecv ({0})'.format(args)

        return code

class MPI_comm_isend(MPI):
    """
    Represents the MPI_isend statement.
    MPI_isend syntax is
    `MPI_ISEND (data, count, datatype, dest, tag, comm)`

    data:
        initial address of send buffer (choice) [IN]
    count:
        number of elements in send buffer (non-negative integer) [IN]
    datatype:
        datatype of each send buffer element (handle) [IN]
    dest:
        rank of destination (integer) [IN]
    tag:
        message tag (integer) [IN]
    comm:
        communicator (handle) [IN]
    request:
        communication request [OUT]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_isend
    >>> n = Variable('int', 'n')
    >>> x = Variable('double', 'x', rank=2, shape=(n,2), allocatable=True)
    >>> dest = Variable('int', 'dest')
    >>> tag  = Variable('int', 'tag')
    >>> requests = Variable('int', 'requests', rank=1, shape=4, allocatable=True)
    >>> comm = MPI_comm_world()
    >>> MPI_comm_isend(x, dest, tag, requests, comm)
    MPI_isend (x, 2*n, MPI_DOUBLE, dest, tag, mpi_comm_world, requests, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_isend, cls).__new__(cls, *args, **options)

    @property
    def data(self):
        return self.args[0]

    @property
    def dest(self):
        return self.args[1]

    @property
    def tag(self):
        return self.args[2]

    @property
    def request(self):
        return self.args[3]

    @property
    def comm(self):
        return self.args[4]

    @property
    def count(self):
        return get_shape(self.data)

    @property
    def datatype(self):
        return mpi_datatype(self.data.dtype)

    def _sympystr(self, printer):
        sstr = printer.doprint

        data    = self.data
        count   = self.count
        dtype   = self.datatype
        dest    = self.dest
        tag     = self.tag
        comm    = self.comm
        request = self.request
        ierr    = MPI_ERROR

        args = (data, count, dtype, dest, tag, comm, request, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_isend ({0})'.format(args)
        return code

class MPI_comm_sendrecv(MPI):
    """
    Represents the MPI_sendrecv statement.
    MPI_sendrecv syntax is
    `MPI_SENDRECV(senddata, sendcount, sendtype, dest, sendtag, recvdata, recvcount, recvtype, source, recvtag, comm, istatus, ierr)`

    senddata:
        initial address of send buffer (choice) [IN]

    sendcount:
        number of elements in send buffer (non-negative integer) [IN]

    senddatatype:
        datatype of each receive buffer element (handle) [IN]

    dest:
        rank of destination (integer) [IN]

    sendtag:
        message tag (integer) [IN]

    recvdata:
        initial address of receive buffer (choice) [OUT]

    recvcount:
        number of elements in receive buffer (non-negative integer) [IN]

    recvdatatype:
        datatype of each send buffer element (handle) [IN]

    source:
        rank of source or MPI_ANY_SOURCE (integer) [IN]

    recvtag:
        message tag or MPI_ANY_TAG (integer) [IN]

    comm:
        communicator (handle) [IN]

    status:
        status object (Status) [OUT]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_sendrecv
    >>> n = Variable('int', 'n')
    >>> x = Variable('double', 'x', rank=2, shape=(n,2), allocatable=True)
    >>> y = Variable('double', 'y', rank=2, shape=(n,2), allocatable=True)
    >>> source = Variable('int', 'source')
    >>> dest   = Variable('int', 'dest')
    >>> sendtag  = Variable('int', 'sendtag')
    >>> recvtag  = Variable('int', 'recvtag')
    >>> comm = MPI_comm_world()
    >>> MPI_comm_sendrecv(x, dest, sendtag, y, source, recvtag, comm)
    MPI_sendrecv (x, 2*n, MPI_DOUBLE, dest, sendtag, y, 2*n, MPI_DOUBLE, source, recvtag, mpi_comm_world, i_mpi_status, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_sendrecv, cls).__new__(cls, *args, **options)

    @property
    def senddata(self):
        return self.args[0]

    @property
    def dest(self):
        return self.args[1]

    @property
    def sendtag(self):
        return self.args[2]

    @property
    def recvdata(self):
        return self.args[3]

    @property
    def source(self):
        return self.args[4]

    @property
    def recvtag(self):
        return self.args[5]

    @property
    def comm(self):
        return self.args[6]

    @property
    def sendcount(self):
        return get_shape(self.senddata)

    @property
    def recvcount(self):
        return get_shape(self.recvdata)

    @property
    def senddatatype(self):
        return mpi_datatype(self.senddata.dtype)

    @property
    def recvdatatype(self):
        return mpi_datatype(self.recvdata.dtype)

    def _sympystr(self, printer):
        sstr = printer.doprint

        senddata  = self.senddata
        recvdata  = self.recvdata
        sendcount = self.sendcount
        recvcount = self.recvcount
        sendtype  = self.senddatatype
        recvtype  = self.recvdatatype
        dest      = self.dest
        source    = self.source
        sendtag   = self.sendtag
        recvtag   = self.recvtag
        comm      = self.comm
        ierr      = MPI_ERROR
        istatus   = MPI_STATUS

        args = (senddata, sendcount, sendtype, dest,   sendtag, \
                recvdata, recvcount, recvtype, source, recvtag, \
                comm, istatus, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_sendrecv ({0})'.format(args)
        return code

class MPI_comm_sendrecv_replace(MPI):
    """
    Represents the MPI_sendrecv_replace statement.
    MPI_sendrecv_replace syntax is
    `MPI_SENDRECV_REPLACE(senddata, sendcount, sendtype, dest, sendtag, source, recvtag, comm, istatus, ierr)`

    senddata:
        initial address of send buffer (choice) [IN]

    sendcount:
        number of elements in send buffer (non-negative integer) [IN]

    senddatatype:
        datatype of each receive buffer element (handle) [IN]

    dest:
        rank of destination (integer) [IN]

    sendtag:
        message tag (integer) [IN]

    source:
        rank of source or MPI_ANY_SOURCE (integer) [IN]

    recvtag:
        message tag or MPI_ANY_TAG (integer) [IN]

    comm:
        communicator (handle) [IN]

    status:
        status object (Status) [OUT]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_sendrecv_replace
    >>> n = Variable('int', 'n')
    >>> x = Variable('double', 'x', rank=2, shape=(n,2), allocatable=True)
    >>> source = Variable('int', 'source')
    >>> dest   = Variable('int', 'dest')
    >>> sendtag  = Variable('int', 'sendtag')
    >>> recvtag  = Variable('int', 'recvtag')
    >>> comm = MPI_comm_world()
    >>> MPI_comm_sendrecv_replace(x, dest, sendtag, source, recvtag, comm)
    MPI_sendrecv_replace (x, 2*n, MPI_DOUBLE, dest, sendtag, source, recvtag, mpi_comm_world, i_mpi_status, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_sendrecv_replace, cls).__new__(cls, *args, **options)

    @property
    def data(self):
        return self.args[0]

    @property
    def dest(self):
        return self.args[1]

    @property
    def sendtag(self):
        return self.args[2]

    @property
    def source(self):
        return self.args[3]

    @property
    def recvtag(self):
        return self.args[4]

    @property
    def comm(self):
        return self.args[5]

    @property
    def count(self):
        return get_shape(self.data)

    @property
    def datatype(self):
        return mpi_datatype(self.data.dtype)

    def _sympystr(self, printer):
        sstr = printer.doprint

        data      = self.data
        count     = self.count
        dtype     = self.datatype
        dest      = self.dest
        source    = self.source
        sendtag   = self.sendtag
        recvtag   = self.recvtag
        comm      = self.comm
        ierr      = MPI_ERROR
        istatus   = MPI_STATUS

        args = (data, count, dtype, dest, sendtag, source, recvtag, \
                comm, istatus, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_sendrecv_replace ({0})'.format(args)
        return code

class MPI_waitall(MPI):
    """
    Represents the MPI_waitall statement.
    MPI_waitall syntax is
    `MPI_WAITALL (count, reqs, statuts)`

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_waitall
    >>> from pyccel.parallel.mpi import MPI_status_type
    >>> requests = Variable('int', 'requests', rank=1, shape=4, allocatable=True)
    >>> mpi_status_size = MPI_status_size()
    >>> stats = Variable('int', 'stats', rank=1, shape=(mpi_status_size,4), allocatable=True)
    >>> MPI_waitall(requests, stats)
    MPI_waitall (4, requests, stats, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_waitall, cls).__new__(cls, *args, **options)

    @property
    def requests(self):
        return self.args[0]

    @property
    def status(self):
        return self.args[1]

    @property
    def count(self):
        return get_shape(self.requests)

    def _sympystr(self, printer):
        sstr = printer.doprint

        requests = self.requests
        count    = self.count
        status   = self.status
        ierr    = MPI_ERROR

        args = (count, requests, status, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_waitall ({0})'.format(args)
        return code

##########################################################
#                  Synchronization
##########################################################
class MPI_comm_barrier(MPI):
    """
    Represents the size of a given communicator.

    Examples

    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_barrier
    >>> comm = MPI_comm_world()
    >>> MPI_comm_barrier(comm)
    mpi_comm_world.barrier
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_barrier, cls).__new__(cls, *args, **options)

    @property
    def comm(self):
        return self.args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{0}.{1}'.format(sstr(self.comm), 'barrier')

class MPI_comm_bcast(MPI):
    """
    Represents the MPI_bcast statement.
    MPI_bcast syntax is
    `MPI_BCAST(data, count, datatype, root, comm)`

    data:
        initial address of send buffer (choice) [IN]
    count:
        number of elements in send buffer (non-negative integer) [IN]
    datatype:
        datatype of each send buffer element (handle) [IN]
    root:
        rank of broadcast root (integer)
    comm:
        communicator (handle) [IN]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_bcast
    >>> n = Variable('int', 'n')
    >>> x = Variable('double', 'x', rank=2, shape=(n,2), allocatable=True)
    >>> root = Variable('int', 'root')
    >>> comm = MPI_comm_world()
    >>> MPI_comm_bcast(x, root, comm)
    MPI_bcast (x, 2*n, MPI_DOUBLE, root, mpi_comm_world, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_bcast, cls).__new__(cls, *args, **options)

    @property
    def data(self):
        return self.args[0]

    @property
    def root(self):
        return self.args[1]

    @property
    def comm(self):
        return self.args[2]

    @property
    def count(self):
        return get_shape(self.data)

    @property
    def datatype(self):
        return mpi_datatype(self.data.dtype)

    def _sympystr(self, printer):
        sstr = printer.doprint

        data  = self.data
        count = self.count
        dtype = self.datatype
        root  = self.root
        comm  = self.comm
        ierr  = MPI_ERROR

        args = (data, count, dtype, root, comm, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_bcast ({0})'.format(args)
        return code

class MPI_comm_scatter(MPI):
    """
    Represents the MPI_scatter statement.
    MPI_scatter syntax is
    `MPI_SCATTER(senddata, sendcount, sendtype, recvdata, recvcount, recvtype,
    root, comm, ierr)`

    Note that we use sendcount = recvcount for the moment.

    senddata:
        initial address of send buffer (choice) [IN]

    sendcount:
        number of elements in send buffer (non-negative integer) [IN]

    senddatatype:
        datatype of each receive buffer element (handle) [IN]

    recvdata:
        initial address of receive buffer (choice) [OUT]

    recvcount:
        number of elements in receive buffer (non-negative integer) [IN]

    recvdatatype:
        datatype of each send buffer element (handle) [IN]

    root:
        rank of broadcast root (integer)

    comm:
        communicator (handle) [IN]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_scatter
    >>> n = Variable('int', 'n')
    >>> x = Variable('double', 'x', rank=2, shape=(n,2), allocatable=True)
    >>> y = Variable('double', 'y', rank=2, shape=(n,2), allocatable=True)
    >>> root   = Variable('int', 'root')
    >>> comm = MPI_comm_world()
    >>> MPI_comm_scatter(x, y, root, comm)
    MPI_scatter (x, 2*n, MPI_DOUBLE, y, 2*n, MPI_DOUBLE, root, mpi_comm_world, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_scatter, cls).__new__(cls, *args, **options)

    @property
    def senddata(self):
        return self.args[0]

    @property
    def recvdata(self):
        return self.args[1]

    @property
    def root(self):
        return self.args[2]

    @property
    def comm(self):
        return self.args[3]

    @property
    def sendcount(self):
        # sendcount = recvcount
        return get_shape(self.recvdata)

    @property
    def recvcount(self):
        return get_shape(self.recvdata)

    @property
    def senddatatype(self):
        return mpi_datatype(self.senddata.dtype)

    @property
    def recvdatatype(self):
        return mpi_datatype(self.recvdata.dtype)

    def _sympystr(self, printer):
        sstr = printer.doprint

        senddata  = self.senddata
        recvdata  = self.recvdata
        sendcount = self.sendcount
        recvcount = self.recvcount
        sendtype  = self.senddatatype
        recvtype  = self.recvdatatype
        root      = self.root
        comm      = self.comm
        ierr      = MPI_ERROR

        args = (senddata, sendcount, sendtype, recvdata, recvcount, recvtype, \
                root, comm, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_scatter ({0})'.format(args)
        return code

class MPI_comm_gather(MPI):
    """
    Represents the MPI_gather statement.
    MPI_gather syntax is
    `MPI_GATHER(senddata, sendcount, sendtype, recvdata, recvcount, recvtype, root, comm)`

    Note that we use recvcount = sendcount for the moment.

    senddata:
        initial address of send buffer (choice) [IN]

    sendcount:
        number of elements in send buffer (non-negative integer) [IN]

    senddatatype:
        datatype of each receive buffer element (handle) [IN]

    recvdata:
        initial address of receive buffer (choice) [OUT]

    recvcount:
        number of elements in receive buffer (non-negative integer) [IN]

    recvdatatype:
        datatype of each send buffer element (handle) [IN]

    root:
        rank of broadcast root (integer)

    comm:
        communicator (handle) [IN]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_gather
    >>> n = Variable('int', 'n')
    >>> x = Variable('double', 'x', rank=2, shape=(n,2), allocatable=True)
    >>> y = Variable('double', 'y', rank=2, shape=(n,2), allocatable=True)
    >>> root   = Variable('int', 'root')
    >>> comm = MPI_comm_world()
    >>> MPI_comm_gather(x, y, root, comm)
    MPI_gather (x, 2*n, MPI_DOUBLE, y, 2*n, MPI_DOUBLE, root, mpi_comm_world, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_gather, cls).__new__(cls, *args, **options)

    @property
    def senddata(self):
        return self.args[0]

    @property
    def recvdata(self):
        return self.args[1]

    @property
    def root(self):
        return self.args[2]

    @property
    def comm(self):
        return self.args[3]

    @property
    def sendcount(self):
        return get_shape(self.senddata)

    @property
    def recvcount(self):
        return get_shape(self.senddata)

    @property
    def senddatatype(self):
        return mpi_datatype(self.senddata.dtype)

    @property
    def recvdatatype(self):
        return mpi_datatype(self.recvdata.dtype)

    def _sympystr(self, printer):
        sstr = printer.doprint

        senddata  = self.senddata
        recvdata  = self.recvdata
        sendcount = self.sendcount
        recvcount = self.recvcount
        sendtype  = self.senddatatype
        recvtype  = self.recvdatatype
        root      = self.root
        comm      = self.comm
        ierr      = MPI_ERROR

        args = (senddata, sendcount, sendtype, recvdata, recvcount, recvtype, \
                root, comm, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_gather ({0})'.format(args)
        return code

class MPI_comm_allgather(MPI):
    """
    Represents the MPI_allgather statement.
    MPI_allgather syntax is
    `MPI_ALLGATHER(senddata, sendcount, sendtype, recvdata, recvcount, recvtype, comm)`

    Note that we use recvcount = sendcount for the moment.

    senddata:
        initial address of send buffer (choice) [IN]

    sendcount:
        number of elements in send buffer (non-negative integer) [IN]

    senddatatype:
        datatype of each receive buffer element (handle) [IN]

    recvdata:
        initial address of receive buffer (choice) [OUT]

    recvcount:
        number of elements in receive buffer (non-negative integer) [IN]

    recvdatatype:
        datatype of each send buffer element (handle) [IN]

    comm:
        communicator (handle) [IN]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_allgather
    >>> n = Variable('int', 'n')
    >>> x = Variable('double', 'x', rank=2, shape=(n,2), allocatable=True)
    >>> y = Variable('double', 'y', rank=2, shape=(n,2), allocatable=True)
    >>> comm = MPI_comm_world()
    >>> MPI_comm_allgather(x, y, comm)
    MPI_allgather (x, 2*n, MPI_DOUBLE, y, 2*n, MPI_DOUBLE, mpi_comm_world, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_allgather, cls).__new__(cls, *args, **options)

    @property
    def senddata(self):
        return self.args[0]

    @property
    def recvdata(self):
        return self.args[1]

    @property
    def comm(self):
        return self.args[2]

    @property
    def sendcount(self):
        return get_shape(self.senddata)

    @property
    def recvcount(self):
        return get_shape(self.senddata)

    @property
    def senddatatype(self):
        return mpi_datatype(self.senddata.dtype)

    @property
    def recvdatatype(self):
        return mpi_datatype(self.recvdata.dtype)

    def _sympystr(self, printer):
        sstr = printer.doprint

        senddata  = self.senddata
        recvdata  = self.recvdata
        sendcount = self.sendcount
        recvcount = self.recvcount
        sendtype  = self.senddatatype
        recvtype  = self.recvdatatype
        comm      = self.comm
        ierr      = MPI_ERROR

        args = (senddata, sendcount, sendtype, recvdata, recvcount, recvtype, \
                comm, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_allgather ({0})'.format(args)
        return code

class MPI_comm_alltoall(MPI):
    """
    Represents the MPI_alltoall statement.
    MPI_alltoall syntax is
    `MPI_ALLTOALL(senddata, sendcount, sendtype, recvdata, recvcount, recvtype, comm)`

    Note that we use sendcount = recvcount = count for the moment.

    senddata:
        initial address of send buffer (choice) [IN]

    sendcount:
        number of elements in send buffer (non-negative integer) [IN]

    senddatatype:
        datatype of each receive buffer element (handle) [IN]

    recvdata:
        initial address of receive buffer (choice) [OUT]

    recvcount:
        number of elements in receive buffer (non-negative integer) [IN]

    recvdatatype:
        datatype of each send buffer element (handle) [IN]

    comm:
        communicator (handle) [IN]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_alltoall
    >>> n = Variable('int', 'n')
    >>> x = Variable('double', 'x', rank=2, shape=(n,2), allocatable=True)
    >>> y = Variable('double', 'y', rank=2, shape=(n,2), allocatable=True)
    >>> count = Variable('int', 'count')
    >>> comm = MPI_comm_world()
    >>> MPI_comm_alltoall(x, y, count, comm)
    MPI_alltoall (x, count, MPI_DOUBLE, y, count, MPI_DOUBLE, mpi_comm_world, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_alltoall, cls).__new__(cls, *args, **options)

    @property
    def senddata(self):
        return self.args[0]

    @property
    def recvdata(self):
        return self.args[1]

    @property
    def count(self):
        return self.args[2]

    @property
    def comm(self):
        return self.args[3]

    @property
    def sendcount(self):
        return self.count

    @property
    def recvcount(self):
        return self.count

    @property
    def senddatatype(self):
        return mpi_datatype(self.senddata.dtype)

    @property
    def recvdatatype(self):
        return mpi_datatype(self.recvdata.dtype)

    def _sympystr(self, printer):
        sstr = printer.doprint

        senddata  = self.senddata
        recvdata  = self.recvdata
        sendcount = self.sendcount
        recvcount = self.recvcount
        sendtype  = self.senddatatype
        recvtype  = self.recvdatatype
        comm      = self.comm
        ierr      = MPI_ERROR

        args = (senddata, sendcount, sendtype, recvdata, recvcount, recvtype, \
                comm, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_alltoall ({0})'.format(args)
        return code

class MPI_comm_reduce(MPI):
    """
    Represents the MPI_reduce statement.
    MPI_reduce syntax is
    `MPI_REDUCE(senddata, recvdata, count, datatype, op, root, comm)`

    senddata:
        initial address of send buffer (choice) [IN]

    recvdata:
        initial address of receive buffer (choice) [OUT]

    count:
        number of elements in send buffer (non-negative integer) [IN]

    datatype:
        datatype of each receive buffer element (handle) [IN]

    op:
        reduce operation (handle)

    root:
        rank of broadcast root (integer)

    comm:
        communicator (handle) [IN]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_reduce
    >>> from pyccel.parallel.mpi import MPI_SUM
    >>> n = Variable('int', 'n')
    >>> x = Variable('double', 'x', rank=2, shape=(n,2), allocatable=True)
    >>> y = Variable('double', 'y', rank=2, shape=(n,2), allocatable=True)
    >>> root   = Variable('int', 'root')
    >>> comm = MPI_comm_world()
    >>> MPI_comm_reduce(x, y, MPI_SUM(), root, comm)
    MPI_reduce (x, y, 2*n, MPI_DOUBLE, MPI_SUM, root, mpi_comm_world, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        args = list(args)
        op = args[2]
        if not isinstance(op, (str, MPI_Operation)):
            raise TypeError('Expecting a string or MPI_Operation for args[2]')

        if isinstance(op, str):
            args[2] = mpi_operation(op)

        return super(MPI_comm_reduce, cls).__new__(cls, *args, **options)

    @property
    def senddata(self):
        return self.args[0]

    @property
    def recvdata(self):
        return self.args[1]

    @property
    def op(self):
        return self.args[2]

    @property
    def root(self):
        return self.args[3]

    @property
    def comm(self):
        return self.args[4]

    @property
    def count(self):
        return get_shape(self.senddata)

    @property
    def datatype(self):
        return mpi_datatype(self.senddata.dtype)

    def _sympystr(self, printer):
        sstr = printer.doprint

        senddata = self.senddata
        recvdata = self.recvdata
        count    = self.count
        dtype    = self.datatype
        op       = self.op
        root     = self.root
        comm     = self.comm
        ierr     = MPI_ERROR

        args = (senddata, recvdata, count, dtype, \
                op, root, comm, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_reduce ({0})'.format(args)
        return code

class MPI_comm_allreduce(MPI):
    """
    Represents the MPI_allreduce statement.
    MPI_allreduce syntax is
    `MPI_ALLREDUCE(senddata, recvdata, count, datatype, op, comm)`

    senddata:
        initial address of send buffer (choice) [IN]

    recvdata:
        initial address of receive buffer (choice) [OUT]

    count:
        number of elements in send buffer (non-negative integer) [IN]

    datatype:
        datatype of each receive buffer element (handle) [IN]

    op:
        reduce operation (handle)

    comm:
        communicator (handle) [IN]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_allreduce
    >>> from pyccel.parallel.mpi import MPI_SUM
    >>> n = Variable('int', 'n')
    >>> x = Variable('double', 'x', rank=2, shape=(n,2), allocatable=True)
    >>> y = Variable('double', 'y', rank=2, shape=(n,2), allocatable=True)
    >>> comm = MPI_comm_world()
    >>> MPI_comm_allreduce(x, y, MPI_SUM(), comm)
    MPI_allreduce (x, y, 2*n, MPI_DOUBLE, MPI_SUM, mpi_comm_world, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        args = list(args)
        op = args[2]
        if not isinstance(op, (str, MPI_Operation)):
            raise TypeError('Expecting a string or MPI_Operation for args[2]')

        if isinstance(op, str):
            args[2] = mpi_operation(op)

        return super(MPI_comm_allreduce, cls).__new__(cls, *args, **options)

    @property
    def senddata(self):
        return self.args[0]

    @property
    def recvdata(self):
        return self.args[1]

    @property
    def op(self):
        return self.args[2]

    @property
    def comm(self):
        return self.args[3]

    @property
    def count(self):
        return get_shape(self.senddata)

    @property
    def datatype(self):
        return mpi_datatype(self.senddata.dtype)

    def _sympystr(self, printer):
        sstr = printer.doprint

        senddata = self.senddata
        recvdata = self.recvdata
        count    = self.count
        dtype    = self.datatype
        op       = self.op
        comm     = self.comm
        ierr     = MPI_ERROR

        args = (senddata, recvdata, count, dtype, \
                op, comm, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_allreduce ({0})'.format(args)
        return code
##########################################################

##########################################################
#                   Communicators
##########################################################
class MPI_comm_split(MPI):
    """
    Represents the MPI_split statement.
    MPI_comm_split syntax is
    `MPI_COMM_SPLIT(comm, color, key, newcomm)`

    color:
        control of subset assignment (integer) [IN]

    key:
        control of rank assigment (integer) [IN]

    comm:
        communicator (handle) [IN]

    newcomm:
        newcomm new communicator (handle) [OUT]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm, MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_split
    >>> color = Variable('int', 'color')
    >>> key   = Variable('int', 'key')
    >>> comm  = MPI_comm_world()
    >>> newcomm = MPI_comm('newcomm')
    >>> MPI_comm_split(color, key, newcomm, comm)
    MPI_comm_split (mpi_comm_world, color, key, newcomm, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_split, cls).__new__(cls, *args, **options)

    @property
    def color(self):
        return self.args[0]

    @property
    def key(self):
        return self.args[1]

    @property
    def newcomm(self):
        return self.args[2]

    @property
    def comm(self):
        return self.args[3]

    def _sympystr(self, printer):
        sstr = printer.doprint

        color   = self.color
        key     = self.key
        comm    = self.comm
        newcomm = self.newcomm
        ierr    = MPI_ERROR

        args = (comm, color, key, newcomm, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_comm_split ({0})'.format(args)
        return code

class MPI_comm_free(MPI):
    """
    Represents the MPI_comm_free statement.
    MPI_comm_free syntax is
    `MPI_COMM_FREE(comm)`

    comm:
        communicator (handle) [IN]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm, MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_split
    >>> color = Variable('int', 'color')
    >>> key = Variable('int', 'key')
    >>> comm = MPI_comm_world()
    >>> newcomm = MPI_comm('newcomm')
    >>> MPI_comm_split(comm, color, key, newcomm)
    >>> MPI_free(newcomm)
    MPI_comm_free (newcomm, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_free, cls).__new__(cls, *args, **options)

    @property
    def comm(self):
        return self.args[0]

    def _sympystr(self, printer):
        sstr = printer.doprint

        comm    = self.comm
        ierr    = MPI_ERROR

        args = (comm, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_comm_free ({0})'.format(args)
        return code

##########################################################

##########################################################
#                  Topologies
##########################################################
class MPI_comm_cart_create(MPI):
    """
    Represents the MPI_cart_create statement.
    MPI_cart_create syntax is
    `MPI_CART_CREATE(comm, ndims, dims, periods, reorder, newcomm)`

    comm:
        input communicator (handle) [IN]

    ndims:
        number of dimensions of Cartesian grid (integer) [IN]

    dims:
        integer array of size ndims specifying the number of processes in each dimension [IN]

    periods:
        logical array of size ndims specifying whether the grid is periodic (true) or not (false) in each dimension [IN]

    reorder:
        ranking may be reordered (true) or not (false) (logical) [IN]

    newcomm:
        communicator with new Cartesian topology (handle) [OUT]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm, MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_cart_create
    >>> n = Variable('int', 'n')
    >>> dims    = Variable('int',     'dims', rank=1, shape=n, allocatable=True)
    >>> periods = Variable('bool', 'periods', rank=1, shape=n, allocatable=True)
    >>> reorder = Variable('bool', 'reorder')
    >>> comm  = MPI_comm_world()
    >>> newcomm = MPI_comm('newcomm')
    >>> MPI_comm_cart_create(dims, periods, reorder, newcomm, comm)
    MPI_cart_create (mpi_comm_world, n, dims, periods, reorder, newcomm, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_cart_create, cls).__new__(cls, *args, **options)

    @property
    def dims(self):
        return self.args[0]

    @property
    def periods(self):
        return self.args[1]

    @property
    def reorder(self):
        return self.args[2]

    @property
    def newcomm(self):
        return self.args[3]

    @property
    def comm(self):
        return self.args[4]

    @property
    def ndims(self):
        return get_shape(self.dims)

    def _sympystr(self, printer):
        sstr = printer.doprint

        ndims   = self.ndims
        dims    = self.dims
        periods = self.periods
        reorder = self.reorder
        comm    = self.comm
        newcomm = self.newcomm
        ierr    = MPI_ERROR

        args = (comm, ndims, dims, periods, reorder, newcomm, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_cart_create ({0})'.format(args)
        return code

class MPI_comm_cart_coords(MPI):
    """
    Represents the MPI_cart_coords statement.
    MPI_cart_coords syntax is
    `MPI_CART_COORDS(comm, rank, maxdims, coords)`

    comm:
        communicator with Cartesian structure (handle) [IN]

    rank:
        rank of a process within group of comm (integer) [IN]

    maxdims:
        length of vector coords in the calling program (integer) [IN]

    coords: integer array (of size ndims) containing the Cartesian coordinates of specified process (array of integers) [OUT]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm, MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_cart_coords
    >>> n = Variable('int', 'n')
    >>> coords = Variable('int', 'coords', rank=1, shape=n, allocatable=True)
    >>> rank = Variable('int', 'rank')
    >>> comm  = MPI_comm_world()
    >>> MPI_comm_cart_coords(rank, coords, comm)
    MPI_cart_coords (mpi_comm_world, rank, n, coords, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_cart_coords, cls).__new__(cls, *args, **options)

    @property
    def rank(self):
        return self.args[0]

    @property
    def coords(self):
        return self.args[1]

    @property
    def comm(self):
        return self.args[2]

    @property
    def ndims(self):
        return get_shape(self.coords)

    def _sympystr(self, printer):
        sstr = printer.doprint

        rank    = self.rank
        ndims   = self.ndims
        coords  = self.coords
        comm    = self.comm
        ierr    = MPI_ERROR

        args = (comm, rank, ndims, coords, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_cart_coords ({0})'.format(args)
        return code

class MPI_comm_cart_shift(MPI):
    """
    Represents the MPI_cart_shift statement.
    MPI_cart_shift syntax is
    `MPI_CART_SHIFT(comm, direction, disp, rank_source, rank_dest)`

    comm:
        communicator with Cartesian structure (handle) [IN]

    direction:
        coordinate dimension of shift (integer) [IN]

    disp:
        displacement (> 0: upwards shift, < 0: downwards shift) (integer)[IN]

    rank_source:
        rank of source process (integer) [OUT]

    rank_dest:
        rank of destination process (integer) [OUT]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm, MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_cart_shift
    >>> direction = Variable('int', 'direction')
    >>> disp = Variable('int', 'disp')
    >>> source = Variable('int', 'source')
    >>> dest = Variable('int', 'dest')
    >>> comm  = MPI_comm_world()
    >>> MPI_comm_cart_shift(direction, disp, source, dest, comm)
    MPI_cart_shift (mpi_comm_world, direction, disp, source, dest, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_cart_shift, cls).__new__(cls, *args, **options)

    @property
    def direction(self):
        return self.args[0]

    @property
    def disp(self):
        return self.args[1]

    @property
    def source(self):
        return self.args[2]

    @property
    def dest(self):
        return self.args[3]

    @property
    def comm(self):
        return self.args[4]

    def _sympystr(self, printer):
        sstr = printer.doprint

        direction = self.direction
        disp      = self.disp
        source    = self.source
        dest      = self.dest
        comm      = self.comm
        ierr      = MPI_ERROR

        args = (comm, direction, disp, source, dest, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_cart_shift ({0})'.format(args)
        return code

class MPI_comm_cart_sub(MPI):
    """
    Represents the MPI_cart_sub statement.
    MPI_cart_create syntax is
    `MPI_CART_SUB(comm, remain_dims, newcomm)`

    comm:
        input communicator (handle) [IN]

    dims:
        the i-th entry of remain_dims specifies whether the i-th dimension
        is kept in the subgrid (true) or is dropped (false) (logical vector) [IN]

    newcomm:
        communicator containing the subgrid that includes the calling process (handle) [OUT]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_comm, MPI_comm_world
    >>> from pyccel.parallel.mpi import MPI_comm_cart_sub
    >>> n = Variable('int', 'n')
    >>> dims = Variable('int', 'dims', rank=1, shape=n, allocatable=True)
    >>> comm  = MPI_comm_world()
    >>> newcomm = MPI_comm('newcomm')
    >>> MPI_comm_cart_sub(dims, newcomm, comm)
    MPI_cart_sub (mpi_comm_world, dims, newcomm, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_cart_sub, cls).__new__(cls, *args, **options)

    @property
    def dims(self):
        return self.args[0]

    @property
    def newcomm(self):
        return self.args[1]

    @property
    def comm(self):
        return self.args[2]

    @property
    def ndims(self):
        return get_shape(self.dims)

    def _sympystr(self, printer):
        sstr = printer.doprint

        dims    = self.dims
        comm    = self.comm
        newcomm = self.newcomm
        ierr    = MPI_ERROR

        args = (comm, dims, newcomm, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_cart_sub ({0})'.format(args)
        return code


# TODO not working yet in pyccel
class MPI_dims_create(MPI):
    """
    Represents the MPI_dims_create statement.
    MPI_comm_free syntax is
    `MPI_DIMS_CREATE(nnodes, ndims, dims)`

    nnodes:
        number of nodes in a grid (integer) [IN]

    ndims:
        number of Cartesian dimensions (integer) [IN]

    dims:
        integer array of size ndims specifying the number
        of nodes in each dimension [INOUT]

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.parallel.mpi import MPI_dims_create
    >>> nnodes = Variable('int', 'nnodes')
    >>> n = Variable('int', 'n')
    >>> dims = Variable('int', 'dims', rank=1, shape=n, allocatable=True)
    >>> MPI_dims_create(nnodes, dims)
    MPI_dims_create (nnodes, n, dims, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_dims_create, cls).__new__(cls, *args, **options)

    @property
    def nnodes(self):
        return self.args[0]

    @property
    def dims(self):
        return self.args[1]

    @property
    def ndims(self):
        return get_shape(self.dims)

    def _sympystr(self, printer):
        sstr = printer.doprint

        nnodes  = self.nnodes
        ndims   = self.ndims
        dims    = self.dims
        ierr    = MPI_ERROR

        args = (nnodes, ndims, dims, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_dims_create ({0})'.format(args)
        return code

##########################################################

##########################################################
#                  Derived types
##########################################################
class MPI_type_contiguous(MPI):
    """
    Represents the contiguous type in mpi.
    MPI_type_contiguous syntax is
    `MPI_TYPE_CONTIGUOUS(count, oldtype, newtype)`

    count:
        number of blocks (non-negative integer) [IN]

    oldtype:
        old datatype (handle) [IN]

    newtype:
        new datatype (handle)  [OUT]

    Examples

    >>> from pyccel.parallel.mpi import MPI_type_vector, MPI_DOUBLE
    >>> count       = 4
    >>> oldtype     = MPI_DOUBLE()
    >>> MPI_type_contiguous('column', count, oldtype)
    MPI_type_contiguous (4, MPI_DOUBLE, column, i_mpi_error)
    MPI_type_commit (column, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_type_contiguous, cls).__new__(cls, *args, **options)

    @property
    def newtype(self):
        return self.args[0]

    @property
    def count(self):
        return self.args[1]

    @property
    def oldtype(self):
        return self.args[2]

    def _sympystr(self, printer):
        sstr = printer.doprint

        count       = self.count
        oldtype     = self.oldtype
        newtype     = self.newtype
        ierr        = MPI_ERROR

        args = (count, oldtype, newtype, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_type_contiguous ({0})'.format(args)

        commit = 'MPI_type_commit ({0}, {1})'.format(newtype, ierr)

        code = '{0}\n{1}'.format(code, commit)

        return code

class MPI_type_vector(MPI):
    """
    Represents the vector type in mpi.
    MPI_type_vector syntax is
    `MPI_TYPE_VECTOR(count, blocklength, stride, oldtype, newtype)`

    count:
        number of blocks (non-negative integer) [IN]

    blocklength:
        number of elements in each block (non-negative integer) [IN]

    stride:
        number of elements between start of each block (integer) [IN]

    oldtype:
        old datatype (handle) [IN]

    newtype:
        new datatype (handle)  [OUT]

    Examples

    >>> from pyccel.parallel.mpi import MPI_type_vector, MPI_DOUBLE
    >>> count       = 4
    >>> blocklength = 1
    >>> stride      = 16
    >>> oldtype     = MPI_DOUBLE()
    >>> MPI_type_vector('line', count, blocklength, stride, oldtype)
    MPI_type_vector (4, 1, 16, MPI_DOUBLE, line, i_mpi_error)
    MPI_type_commit (line, i_mpi_error)
    """
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_type_vector, cls).__new__(cls, *args, **options)

    @property
    def newtype(self):
        return self.args[0]

    @property
    def count(self):
        return self.args[1]

    @property
    def blocklength(self) :
        return self.args[2]

    @property
    def stride(self):
        return self.args[3]

    @property
    def oldtype(self):
        return self.args[4]

    def _sympystr(self, printer):
        sstr = printer.doprint

        count       = self.count
        blocklength = self.blocklength
        stride      = self.stride
        oldtype     = self.oldtype
        newtype     = self.newtype
        ierr        = MPI_ERROR

        args = (count, blocklength, stride, oldtype, newtype, ierr)
        args  = ', '.join('{0}'.format(sstr(a)) for a in args)
        code = 'MPI_type_vector ({0})'.format(args)

        commit = 'MPI_type_commit ({0}, {1})'.format(newtype, ierr)

        code = '{0}\n{1}'.format(code, commit)

        return code
##########################################################

##########################################################
# The following classes are to
# provide user friendly support of MPI
##########################################################
class MPI_Tensor(MPI, Block, Tensor):
    """
    Represents a Tensor object using MPI.

    Examples

    >>> from pyccel.types.ast import Variable
    >>> from pyccel.types.ast import Range, Tensor
    >>> from pyccel.parallel.mpi import MPI_Tensor
    >>> from sympy import Symbol
    >>> s1 = Variable('int', 's1')
    >>> s2 = Variable('int', 's2')
    >>> e1 = Variable('int', 'e1')
    >>> e2 = Variable('int', 'e2')
    >>> r1 = Range(s1, e1, 1)
    >>> r2 = Range(s2, e2, 1)
    >>> tensor = Tensor(r1, r2)
    >>> from pyccel.parallel.mpi import MPI_comm_world
    >>> comm = MPI_comm_world()
    >>> MPI_Tensor(tensor, comm)
    """
    is_integer = True

    def __new__(cls, tensor, \
                comm_parent=None, \
                dims=None, periods=None, reorder=False, \
                disp=1, label=None):
        # ...
        if not isinstance(tensor, Tensor):
            raise TypeError('Expecting a Tensor')
        cls._tensor = tensor
        cls._label  = label
        # ...

        # ...
        def _make_name(n):
            if not label:
                return n
            if len(label) > 0:
                return '{0}_{1}'.format(label, n)
            else:
                return n
        # ...

        # ...
        variables = []
        body      = []
        # ...

        # ... we don't need to append ierr to variables, since it will be
        #     defined in the program
        ierr = MPI_ERROR
        #variables.append(ierr)
        # ...

        # ...
        ndim = Variable('int', _make_name('ndim'))
        stmt = Assign(ndim, tensor.dim)
        variables.append(ndim)
        body.append(stmt)

        cls._ndim = ndim
        # ...

        # ... TODO use MPI_dims_create
        if dims is None:
            dims = (2,2)

        if not isinstance(dims, (list, tuple)):
           raise TypeError('Expecting a tuple or list')

        dims  = list(dims)
        _dims = []
        for a in dims:
            if isinstance(a, int):
                _dims.append(a)
            elif isinstance(a, Variable) and isinstance(a.dtype, NativeInteger):
                _dims.append(a)
            else:
               raise TypeError('Expecting an integer')

        dims = Variable('int', _make_name('dims'), \
                        rank=1, shape=ndim, allocatable=True)
        stmt = Zeros(dims, ndim)
        variables.append(dims)
        body.append(stmt)

        dims = IndexedVariable(dims.name, dtype=dims.dtype, shape=dims.shape)
        for i in range(0, tensor.dim):
            stmt = Assign(dims[i], _dims[i])
            body.append(stmt)

        cls._dims = dims
        # ...

        # ...
        if periods is None:
            periods = (False,False)

        if not isinstance(periods, (list, tuple)):
           raise TypeError('Expecting a tuple or list')

        periods  = list(periods)
        _periods = []
        for a in periods:
            if isinstance(a, bool):
                if a:
                    _periods.append(BooleanTrue())
                else:
                    _periods.append(BooleanFalse())
            elif isinstance(a, Variable) and isinstance(a.dtype, NativeBool):
                _periods.append(a)
            else:
               raise TypeError('Expecting a Boolean')

        periods = Variable('bool', _make_name('periods'), \
                           rank=1, shape=ndim, allocatable=True)
        stmt = Zeros(periods, ndim)
        variables.append(periods)
        body.append(stmt)

        periods = IndexedVariable(periods.name, dtype=periods.dtype)
        for i in range(0, tensor.dim):
            stmt = Assign(periods[i], _periods[i])
            body.append(stmt)

        cls._periods = periods
        # ...

        # ...
        if reorder:
            reorder_val = BooleanTrue()
        else:
            reorder_val = BooleanFalse()

        reorder = Variable('bool', _make_name('reorder'))
        stmt = Assign(reorder, reorder_val)
        variables.append(reorder)
        body.append(stmt)

        cls._reorder = reorder
        # ...

        # ... set the parent comm
        if comm_parent is None:
            comm_parent = MPI_comm_world()
        else:
            if not isinstance(comm_parent, MPI_comm):
                raise TypeError('Expecting a valid MPI communicator')
        cls._comm_parent = comm_parent
        # ...

        # ... create the cart comm
        comm_name = _make_name('comm_cart')
        comm = Variable('int', comm_name, rank=0, cls_base=MPI_comm())
        variables.append(comm)

        comm = MPI_comm(comm_name)
        rhs  = MPI_comm_cart_create(dims, periods, reorder, comm, comm_parent)
        stmt = MPI_Assign(ierr, rhs, strict=False)
        body.append(stmt)

        cls._comm = comm
        # ...

        # ...
        rank_in_cart = Variable('int', _make_name('rank_in_cart'))
        stmt = MPI_Assign(rank_in_cart, MPI_comm_rank(comm))
        variables.append(rank_in_cart)
        body.append(stmt)

        cls._rank_in_cart = rank_in_cart
        # ...

        # ... compute the coordinates of the process
        coords = Variable('int', _make_name('coords'), \
                          rank=1, shape=ndim, allocatable=True)
        stmt = Zeros(coords, ndim)
        variables.append(coords)
        body.append(stmt)

        rhs  = MPI_comm_cart_coords(rank_in_cart, coords, comm)
        stmt = MPI_Assign(ierr, rhs, strict=False)
        body.append(stmt)

        coords = IndexedVariable(coords.name, \
                                 dtype=coords.dtype, \
                                 shape=coords.shape)
        cls._coords = coords
        # ...

        # ... TODO treat disp properly
        neighbor = Variable('int', _make_name('neighbor'), \
                            rank=1, shape=2*ndim, allocatable=True)
        stmt = Zeros(neighbor, 2*ndim)
        variables.append(neighbor)
        body.append(stmt)

        neighbor = IndexedVariable(neighbor.name, dtype=neighbor.dtype)
        cls._neighbor = neighbor

        _map_neighbor = {}
        if tensor.dim == 2:
            north = 0 ; east = 1 ; south = 2 ; west = 3

            # ...
            axis = 0
            rhs  = MPI_comm_cart_shift(axis, disp, \
                                       neighbor[west], neighbor[east], \
                                       comm)
            stmt = MPI_Assign(ierr, rhs, strict=False)
            body.append(stmt)
            # ...

            # ...
            axis = 1
            rhs  = MPI_comm_cart_shift(axis, disp, \
                                       neighbor[south], neighbor[north], \
                                       comm)
            stmt = MPI_Assign(ierr, rhs, strict=False)
            body.append(stmt)
            # ...
        else:
            raise NotImplementedError('Only 2d is available')
        # ...

        # ... compute local ranges
        starts = [r.start for r in tensor.ranges]
        ends   = [r.stop  for r in tensor.ranges]
        steps  = [r.step  for r in tensor.ranges]

        d = {}
        labels = ['x','y','z'][:tensor.dim]
        for i,l in enumerate(labels):
            nn = (ends[i] - starts[i])/steps[i]

            d['s'+l] = (coords[i] * nn) / dims[i]
            d['e'+l] = ((coords[i]+1) * nn) / dims[i]

        ranges = []
        d_var = {}
        for l in labels:
            dd = {}
            for _n in ['s', 'e']:
                n = _n+l
                v    = Variable('int', _make_name(n))
                rhs  = d[n]
                stmt = Assign(v, rhs)
                variables.append(v)
                body.append(stmt)

                dd[n] = v
                d_var[n] = v

            args = [i[1] for i in dd.items()]
            r = Range(*args)
            ranges.append(r)

        cls._ranges = ranges
        # ...

        # ... derived types for communication over boundaries
        ex = d_var['ex']
        sx = d_var['sx']
        ey = d_var['ey']
        sy = d_var['sy']


        # Creation of the type_line derived datatype to exchange points
        # with northern to southern neighbours
        count       = ey-sy+1
        blocklength = 1
        stride      = ex-sx+3
        oldtype     = MPI_DOUBLE()

        line = Variable('int', _make_name('line'))
        variables.append(line)

        rhs = MPI_type_vector(line, count, blocklength, stride, oldtype)
        stmt = MPI_Assign(ierr, rhs, strict=False)
        body.append(stmt)
        # 

        # Creation of the type_column derived datatype to exchange points
        # with western to eastern neighbours
        count   = ex-sx+1
        oldtype = MPI_DOUBLE()

        column = Variable('int', _make_name('column'))
        variables.append(column)

        rhs = MPI_type_contiguous(column, count, oldtype)
        stmt = MPI_Assign(ierr, rhs, strict=False)
        body.append(stmt)
        #

        cls._types_bnd = {}
        cls._types_bnd['line'] = line
        cls._types_bnd['column'] = column
        # ...

        return super(MPI_Tensor, cls).__new__(cls, variables, body)

    @property
    def tensor(self):
        return self._tensor

    @property
    def comm_parent(self):
        return self._comm_parent

    @property
    def comm(self):
        return self._comm

    @property
    def reorder(self):
        return self._reorder

    @property
    def neighbor(self):
        return self._neighbor

    @property
    def coords(self):
        return self._coords

    @property
    def dim(self):
        return self.tensor.dim

    @property
    def ndim(self):
        return self._ndim

    @property
    def dims(self):
        return self._dims

    @property
    def periods(self):
        return self._periods

    @property
    def reorder(self):
        return self._reorder

    @property
    def rank_in_cart(self):
        return self._rank_in_cart

    @property
    def ranges(self):
        return self._ranges

    @property
    def types_bnd(self):
        return self._types_bnd

    @property
    def label(self):
        return self._label

    def free_statements(self):
        """Returns a list of Free ast objects."""
        ls = []

        stmt = MPI_comm_free(self.comm)
        ls.append(stmt)

        for v in self.variables:
            if isinstance(v, Variable):
                if v.allocatable: ls.append(Del(v))

        return ls

    def _sympystr(self, printer):
        sstr = printer.doprint

        variables = self.variables
        body      = self.body
        ierr      = MPI_ERROR

        variables  = ', '.join('{0}'.format(sstr(a)) for a in variables)
        body       = ', '.join('{0}'.format(sstr(a)) for a in body)
        code = 'MPI_Tensor ([{0}], [{1}])'.format(variables, body)
        return code
##########################################################

##########################################################
#             Communication over topologies
##########################################################
class MPI_Communication(MPI):
    """MPI communication action."""
    pass

class MPI_TensorCommunication(MPI_Communication, Block):
    """MPI communication over a MPI_Tensor object."""
    is_integer = True

    def __new__(cls, tensor, variables):
        if not isinstance(tensor, MPI_Tensor):
            raise TypeError('Expecting MPI_Tensor')

        if not iterable(variables):
            raise TypeError('Expecting an iterable of variables')

        # ...
        def _make_name(n):
            label = tensor.label
            if not label:
                return n
            if len(label) > 0:
                return '{0}_{1}'.format(label, n)
            else:
                return n
        # ...

        # ...
        body = []
        local_vars = []
        # ...

        # ... we don't need to append ierr to variables, since it will be
        #     defined in the program
        ierr = MPI_ERROR
        #variables.append(ierr)
        # ...

        # ...
        cls._tensor = tensor
        # ...

        # ...
        starts = [r.start for r in tensor.ranges]
        ends   = [r.stop  for r in tensor.ranges]
        steps  = [r.step  for r in tensor.ranges]

        sx = starts[0] ; sy = starts[1]
        ex = ends[0]  ; ey = ends[1]

        type_line   = tensor.types_bnd['line']
        type_column = tensor.types_bnd['column']

        north = 0 ; east = 1 ; south = 2 ; west = 3
        neighbor = tensor.neighbor

        comm = tensor.comm

        tag = _make_name('tag')
        tag_value = int(str(abs(hash(tag)))[-6:])
        tag_name = '{0}_{1}'.format(tag, str(tag_value))
        tag = Variable('int', tag_name)
        local_vars.append(tag)

        stmt = Assign(tag, tag_value)
        body.append(stmt)
        # ...

        # ... # TODO loop over variables
        u = variables[0]
        rhs = MPI_comm_sendrecv(u, neighbor[north], tag, \
                                u, neighbor[south], tag, comm)
        stmt = MPI_Assign(ierr, rhs, strict=False)
        body.append(stmt)
        # ...


#    !Send to neighbour N and receive from neighbour S
#    CALL MPI_SENDRECV(u(sx, sy), 1,   type_line,           neighbour(N), &
#         tag,  u(ex+1, sy), 1,        type_line,           neighbour(S), &
#         tag, comm2d, status, code)
#
#    !Send to neighbour S and receive from neighbour N
#    CALL MPI_SENDRECV(u(ex, sy), 1,   type_line,           neighbour(S), &
#         tag,  u(sx-1, sy), 1,        type_line,           neighbour(N), &
#         tag, comm2d, status, code)
#
#    !Send to neighbour W  and receive from neighbour E
#    CALL MPI_SENDRECV(u(sx, sy), 1, type_column,           neighbour(W), &
#         tag,  u(sx, ey+1), 1, type_column,                neighbour(E), &
#         tag, comm2d, status, code)
#
#    !Send to neighbour E  and receive from neighbour W
#    CALL MPI_SENDRECV(u(sx, ey), 1, type_column,           neighbour(E), &
#         tag,  u(sx, sy-1), 1, type_column,                neighbour(W), &
#         tag, comm2d, status, code)

        return super(MPI_TensorCommunication, cls).__new__(cls, local_vars, body)

    @property
    def tensor(self):
        return self._tensor

##########################################################

##########################################################
#             useful functions
##########################################################
def mpify(stmt, **options):
    """
    Converts some statements to MPI statments.

    stmt: stmt, list
        statement or a list of statements
    """
    if isinstance(stmt, MPI):
        return stmt
    if isinstance(stmt, Tensor):
        options['label'] = stmt.name
        return MPI_Tensor(stmt, **options)
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
        name        = mpify(stmt.name,        **options)
        arguments   = mpify(stmt.arguments,   **options)
        results     = mpify(stmt.results,     **options)
        body        = mpify(stmt.body,        **options)
        local_vars  = mpify(stmt.local_vars,  **options)
        global_vars = mpify(stmt.global_vars, **options)

        return FunctionDef(name, arguments, results, \
                           body, local_vars, global_vars)
    if isinstance(stmt, ClassDef):
        name        = mpify(stmt.name,        **options)
        attributs   = mpify(stmt.attributs,   **options)
        methods     = mpify(stmt.methods,     **options)
        options     = mpify(stmt.options,     **options)

        return ClassDef(name, attributs, methods, options)
    if isinstance(stmt, Assign):
        if isinstance(stmt.rhs, MPI_Tensor):
            return stmt
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
    if isinstance(stmt, Sync):
        if stmt.master:
            variables = [mpify(a, **options) for a in stmt.variables]
            master  = mpify(stmt.master, **options)
            if isinstance(master, MPI_Tensor):
                return MPI_TensorCommunication(master, variables)

    return stmt
##########################################################

MPI_ERROR   = Variable('int', 'i_mpi_error')
MPI_STATUS  = Variable(MPI_status_type(), 'i_mpi_status')

MPI_COMM_WORLD  = MPI_comm_world()
MPI_STATUS_SIZE = MPI_status_size()
MPI_PROC_NULL   = MPI_proc_null()

# ...
def mpi_definitions(namespace, declarations):
    """Adds MPI functions and constants to the namespace

    namespace: dict
        dictorionary containing all declared variables/functions/classes.

    declarations: dict
        dictorionary containing all declarations.
    """
    # ...
    namespace['mpi_comm_world']  = MPI_COMM_WORLD
    namespace['mpi_status_size'] = MPI_STATUS_SIZE
    namespace['mpi_proc_null']   = MPI_PROC_NULL
    # ...

    # ...
    for i in [MPI_ERROR, MPI_STATUS]:
        namespace[i.name] = i

        dec = MPI_Declare(i.dtype, i)
        declarations[i.name] = dec
    # ...

    # ...
    body        = []
    local_vars  = []
    global_vars = []
    hide        = True
    kind        = 'procedure'
    # ...

    # ...
    args        = []
    datatype    = 'int'
    allocatable = False
    shape       = None
    rank        = 0

    var_name = 'result_%d' % abs(hash(datatype))

    for f_name in ['mpi_init', 'mpi_finalize']:
        var = Variable(datatype, var_name)
        results = [var]

        stmt = FunctionDef(f_name, args, results, \
                           body, local_vars, global_vars, \
                           hide=hide, kind=kind)

        namespace[f_name] = stmt
    # ...

    # ...
    i = np.random.randint(10)
    var_name = 'result_%d' % abs(hash(i))
    err_name = 'error_%d' % abs(hash(i))

    for f_name in ['mpi_comm_size', 'mpi_comm_rank', 'mpi_abort']:
        var = Variable(datatype, var_name)
        err = Variable(datatype, err_name)
        results = [var, err]

        args = [namespace['mpi_comm_world']]
        stmt = FunctionDef(f_name, args, results, \
                           body, local_vars, global_vars, \
                           hide=hide, kind=kind)

        namespace[f_name] = stmt
    # ...

    return namespace, declarations
# ...

