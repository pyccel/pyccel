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
from sympy.core.function import Function
from sympy.core.function import UndefinedFunction

from pyccel.types.ast import DottedName
from pyccel.types.ast import Variable, IndexedVariable, IndexedElement
from pyccel.types.ast import Assign, Declare, AugAssign
from pyccel.types.ast import NativeBool, NativeFloat
from pyccel.types.ast import NativeComplex, NativeDouble, NativeInteger
from pyccel.types.ast import DataType
from pyccel.types.ast import DataTypeFactory
from pyccel.types.ast import Block
from pyccel.types.ast import Range, Tile, Tensor
from pyccel.types.ast import Zeros
from pyccel.types.ast import Ones
from pyccel.types.ast import Comment
from pyccel.types.ast import EmptyLine
from pyccel.types.ast import Print
from pyccel.types.ast import Len
from pyccel.types.ast import Import

from pyccel.types.ast import For, While, If, Del, Sync
from pyccel.types.ast import FunctionDef, ClassDef
from pyccel.types.ast import MethodCall, FunctionCall

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
    elif isinstance(dtype, Variable):
        # TODO add some checks
        return dtype
    else:
        # Pyccel user class
        cls_name = dtype.__class__.__name__
        try:
            type_name = cls_name.split('PyccelDatatype_')[-1]
            return type_name
        except:
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

class MPI_MAX(MPI_Operation):
    _name   = 'MPI_MAX'
    _symbol = 'max'

    def _sympystr(self, printer):
        return 'MPI_MAX'

class MPI_MIN(MPI_Operation):
    _name   = 'MPI_MIN'
    _symbol = 'min'

    def _sympystr(self, printer):
        return 'MPI_MIN'

_op_registry = {'+': MPI_SUM(), \
                '*': MPI_PROD(), \
                'max': MPI_MAX(), \
                'min': MPI_MIN()}


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
        if not isinstance(op, (str, MPI_Operation, UndefinedFunction)):
            raise TypeError('Expecting a string or MPI_Operation for args[2]')

        # needed for 'max' and 'min' cases
        if isinstance(op, UndefinedFunction):
            op = op.__name__

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

# TODO check that op is a valid operation
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
        if not isinstance(op, (str, MPI_Operation, UndefinedFunction)):
            raise TypeError('Expecting a string or MPI_Operation for args[2]')

        # needed for 'max' and 'min' cases
        if isinstance(op, UndefinedFunction):
            op = op.__name__

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
class MPI_Tensor(ClassDef, MPI, Tensor):
    """
    Represents a Tensor object using MPI.

    Examples

    >>> from pyccel.parallel.mpi import MPI_Tensor
    >>> T = MPI_Tensor()
    >>> T.attributs
    (ndim, nnodes, rank, line, column, comm, dims, periods, coords, neighbor, starts, ends, pads)
    """
    is_integer = True
    _instance = '2d'

    def __new__(cls):
        # ...
        options = ['public']
        # ...

        # ...
        imports = [Import('mpi')]
        # ...

        # ... attributs
        ndim     = Variable('int', 'ndim')
        nnodes   = Variable('int', 'nnodes')
        rank     = Variable('int', 'rank')
        line     = Variable('int', 'line')
        column   = Variable('int', 'column')

        comm     = Variable('int', 'comm', cls_base=MPI_comm())

        dims     = Variable('int', 'dims', \
                            rank=1, shape=ndim, allocatable=True)
        periods  = Variable('bool', 'periods', \
                            rank=1, shape=ndim, allocatable=True)
        coords   = Variable('int', 'coords', \
                            rank=1, shape=ndim, allocatable=True)
        neighbor = Variable('int', 'neighbor', \
                            rank=1, shape=2*ndim, allocatable=True)
        # starts & ends to replace sx,ex, ...
        starts   = Variable('int', 'starts', \
                            rank=1, shape=ndim, allocatable=True)
        ends     = Variable('int', 'ends', \
                            rank=1, shape=ndim, allocatable=True)
        pads     = Variable('int', 'pads', \
                            rank=1, shape=ndim, allocatable=True)
        # ...

        # ...
        attributs = [ndim, nnodes, rank, line, column, \
                     comm, dims, periods, coords, neighbor, \
                     starts, ends, pads]
        # ...

        # ...
        methods = [MPI_Tensor_create(), \
                   MPI_Tensor_free(attributs), \
                   MPI_Tensor_communicate()]
        # ...

        return ClassDef.__new__(cls, 'MPI_Tensor', \
                                attributs, methods, \
                                options=options, \
                                imports=imports)

    @property
    def module(self):
        return 'pcl_m_tensor_{0}'.format(self._instance)

    @property
    def dtype(self):
        return 'pcl_t_tensor_{0}'.format(self._instance)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return '{}'.format(sstr(self.name))

    def get_ranges(self, O):
        starts = self.get_attribute(O, 'starts')
        ends   = self.get_attribute(O, 'ends')
        pads   = self.get_attribute(O, 'pads')

        starts = IndexedVariable(starts.name, dtype=starts.dtype)
        ends   = IndexedVariable(ends.name, dtype=ends.dtype)
        pads   = IndexedVariable(pads.name, dtype=pads.dtype)

        ndim = 2 # TODO get it from O
        ranges = []
        for i in range(0, ndim):
            r = Tile(starts[i]-pads[i], ends[i]+pads[i])
            ranges.append(r)
        return Tensor(*ranges)


class MPI_Tensor_create(FunctionDef):
    """
    Represents a Tensor create procedure.

    Examples

    >>> from pyccel.parallel.mpi import MPI_Tensor_create
    >>> T = MPI_Tensor_create()
    >>> T
    self := MPI_Tensor_create(npts, periods, reorder, pads)
    >>> T.print_body()
    """
    def __new__(cls):
        """
        Represents a call to create for MPI tensor.
        """
        # ...
        f_name = '__init__'

        cls._name = f_name
        # ...

        # ...
        body        = []
        local_vars  = []
        global_vars = []
        imports     = [Import('mpi')]
        hide        = False
        kind        = 'procedure'
        cls_name    = '__UNDEFINED__'
        # ...

        # TODO add comm_parent as (optional) argument

        # ... args
        c_name = 'MPI_Tensor'
        alias  = None
        c_dtype = DataTypeFactory(c_name, ("_name"))

        this = Variable(c_dtype(), 'self')

        ndim         = Variable('int', 'ndim')

        arg_npts     = Variable('int', 'npts', \
                                rank=1, shape=ndim, allocatable=False)
        arg_periods  = Variable('bool', 'periods', \
                                rank=1, shape=ndim, allocatable=False)
        arg_reorder  = Variable('bool', 'reorder')
        arg_pads     = Variable('int', 'pads', \
                                rank=1, shape=ndim, allocatable=False)

        args = [this, arg_npts, arg_periods, arg_reorder, arg_pads]
        # ...

        # ... attributs
        ndim     = Variable('int', DottedName('self', 'ndim'))
        nnodes   = Variable('int', DottedName('self', 'nnodes'))
        rank     = Variable('int', DottedName('self', 'rank'))
        line     = Variable('int', DottedName('self', 'line'))
        column   = Variable('int', DottedName('self', 'column'))

        comm     = Variable('int', DottedName('self', 'comm'), cls_base=MPI_comm())

        dims     = Variable('int', DottedName('self', 'dims'), \
                            rank=1, shape=ndim, allocatable=True)
        periods  = Variable('bool', DottedName('self', 'periods'), \
                            rank=1, shape=ndim, allocatable=True)
        coords   = Variable('int', DottedName('self', 'coords'), \
                            rank=1, shape=ndim, allocatable=True)
        neighbor = Variable('int', DottedName('self', 'neighbor'), \
                            rank=1, shape=2*ndim, allocatable=True)
        # starts & ends to replace sx,ex, ...
        starts   = Variable('int', DottedName('self', 'starts'), \
                            rank=1, shape=ndim, allocatable=True)
        ends     = Variable('int', DottedName('self', 'ends'), \
                            rank=1, shape=ndim, allocatable=True)
        pads     = Variable('int', DottedName('self', 'pads'), \
                            rank=1, shape=ndim, allocatable=True)
        # ...

        # ...
        ierr = MPI_ERROR

        local_vars = [ierr]
        # ...

        # ...
        reorder = arg_reorder
        # ...

        # ...
        body.append(Comment('... MPI_Tensor: grid setting'))
        # ...

        # ... TODO sets the right value. now it is equal to 1
        ndim_value = 2 # arg_npts.rank
        body += [Assign(ndim, Len(arg_npts))]
        # ...

        # ...
        body += [Zeros(dims, ndim)]

        # we change the definition to IndexedVariable to use dims as an array
        dims = IndexedVariable(dims.name, dtype=dims.dtype, shape=dims.shape)

        rhs  = MPI_comm_size(MPI_comm_world())
        body += [MPI_Assign(nnodes, rhs)]

        rhs = MPI_dims_create(nnodes, dims)
        body += [MPI_Assign(ierr, rhs, strict=False)]
        # ...

        # ...
        body += [Zeros(periods, ndim)]

        # we change the definition to IndexedVariable to use periods as an array
        body += [Assign(periods, arg_periods)]

        periods     = IndexedVariable(periods.name, dtype=periods.dtype)
        arg_periods = IndexedVariable(arg_periods.name, dtype=arg_periods.dtype)
        # ...

        # ...
        body.append(Comment('...'))
        body.append(EmptyLine())
        body.append(Comment('... MPI_Tensor: cart definition'))
        # ...

        # ... set the parent comm
        comm_parent = MPI_comm_world()
        # ...

        # ... create the cart comm
        comm = MPI_comm('self % comm')
        rhs  = MPI_comm_cart_create(dims, periods, reorder, comm, comm_parent)

        body += [MPI_Assign(ierr, rhs, strict=False)]
        # ...

        # ...
        body += [MPI_Assign(rank, MPI_comm_rank(comm))]
        # ...

        # ...
        body.append(Comment('...'))
        body.append(EmptyLine())
        body.append(Comment('... MPI_Tensor: Neighbors'))
        # ...

        # ... compute the coordinates of the process
        body += [Zeros(coords, ndim)]

        rhs  = MPI_comm_cart_coords(rank, coords, comm)
        body += [MPI_Assign(ierr, rhs, strict=False)]

        coords = IndexedVariable(coords.name, \
                                 dtype=coords.dtype, \
                                 shape=coords.shape)
        # ...

        # ... TODO treat disp properly
        body += [Zeros(neighbor, 2*ndim)]

        neighbor = IndexedVariable(neighbor.name, dtype=neighbor.dtype)

        _map_neighbor = {}
        if ndim_value == 2:
            north = 0 ; east = 1 ; south = 2 ; west = 3

            # TODO sets disp from pads?
            disp = 1

            # ...
            axis = 0
            rhs  = MPI_comm_cart_shift(axis, disp, \
                                       neighbor[north], neighbor[south], \
                                       comm)

            body += [MPI_Assign(ierr, rhs, strict=False)]
            # ...

            # ...
            axis = 1
            rhs  = MPI_comm_cart_shift(axis, disp, \
                                       neighbor[west], neighbor[east], \
                                       comm)

            body += [MPI_Assign(ierr, rhs, strict=False)]
            # ...
        else:
            raise NotImplementedError('Only 2d is available')
        # ...

        # ...
        body.append(Comment('...'))
        body.append(EmptyLine())
        body.append(Comment('... MPI_Tensor: local ranges'))
        # ...

        # ... compute local ranges
        _starts = np.zeros(ndim_value, dtype=int)
        _steps  = np.ones(ndim_value,  dtype=int)
        _ends   = IndexedVariable(arg_npts.name, dtype=arg_npts.dtype)

        d = {}
        labels = ['x','y','z'][:ndim_value]
        for i,l in enumerate(labels):
            nn = (_ends[i] - _starts[i])/_steps[i]

            d['s'+l] = (coords[i] * nn) / dims[i]
            d['e'+l] = ((coords[i]+1) * nn) / dims[i] - 1

        ranges = []
        d_var = {}
        for l in labels:
            dd = {}
            for _n in ['s', 'e']:
                n = _n+l
                v    = Variable('int', n)
                rhs  = d[n]
                stmt = Assign(v, rhs)
                body.append(stmt)
                local_vars.append(v)

                dd[n] = v
                d_var[n] = v

            _args = [i[1] for i in dd.items()]
            r = Tile(*_args)
            ranges.append(r)
        # ...

        # ...
        body.append(If(((d_var['sx'] > 0), [AugAssign(d_var['sx'],'+',1)])))
        body.append(If(((d_var['sy'] > 0), [AugAssign(d_var['sy'],'+',1)])))
        # ...

        # ...
        body.append(Comment('...'))
        body.append(EmptyLine())
        body.append(Comment('... MPI_Tensor: vector types for communication'))
        # ...

        # ... derived types for communication over boundaries
        sx = d_var['sx']
        ex = d_var['ex']
        sy = d_var['sy']
        ey = d_var['ey']

        body += [Zeros(starts, ndim)]
        body += [Zeros(ends, ndim)]

        starts = IndexedVariable(starts.name, dtype=starts.dtype)
        ends   = IndexedVariable(ends.name,   dtype=ends.dtype)

        body += [Assign(starts[0], sx)]
        body += [Assign(ends[0],   ex)]
        body += [Assign(starts[1], sy)]
        body += [Assign(ends[1],   ey)]
        # ...

        # Creation of the type_line derived datatype to exchange points
        # with northern to southern neighbours
        count       = ey-sy+1
        blocklength = 1
        stride      = ex-sx+3
        oldtype     = MPI_DOUBLE()

        rhs = MPI_type_vector(line, count, blocklength, stride, oldtype)
        body += [MPI_Assign(ierr, rhs, strict=False)]
        # 

        # Creation of the type_column derived datatype to exchange points
        # with western to eastern neighbours
        count   = ex-sx+1
        oldtype = MPI_DOUBLE()

        rhs = MPI_type_contiguous(column, count, oldtype)
        body += [MPI_Assign(ierr, rhs, strict=False)]
        #
        # ...

        # ...
        body.append(Comment('...'))
        body.append(EmptyLine())
        body.append(Comment('... MPI_Tensor: ghost cells size'))
        # ...

        # ...
        body += [Zeros(pads, ndim)]

        pads     = IndexedVariable(pads.name, dtype=pads.dtype, shape=pads.shape)
        arg_pads = IndexedVariable(arg_pads.name, \
                                   dtype=arg_pads.dtype, \
                                   shape=arg_pads.shape)
        for i in range(0, ndim_value):
            body += [Assign(pads[i], arg_pads[i])]
        # ...

        # ...
        body.append(Comment('...'))
        body.append(EmptyLine())
        # ...

        # ...
        results = []
        # ...

        return FunctionDef.__new__(cls, f_name, args, results, \
                                   body, local_vars, global_vars, \
                                   hide=hide, \
                                   kind=kind, \
                                   cls_name=cls_name, \
                                   imports=imports)

    @property
    def name(self):
        return self._name

    def _sympystr(self, printer):
        sstr = printer.doprint

        name    = 'MPI_Tensor_{0}'.format(sstr(self.name))
        args    = ', '.join(sstr(i) for i in self.arguments)
        results = ', '.join(sstr(i) for i in self.results)
        return '{0} := {1}({2})'.format(results, name, args)

class MPI_Tensor_communicate(FunctionDef):
    """
    Represents a Tensor communicate procedure.

    Examples

    >>> from pyccel.parallel.mpi import MPI_Tensor_communicate
    >>> T = MPI_Tensor_communicate()
    >>> T
    self := MPI_Tensor_communicate(npts, periods, reorder, pads)
    >>> T.print_body()
    """
    def __new__(cls):
        """
        Represents a call to communicate for MPI tensor.
        """
        # ...
        f_name = 'communicate'

        cls._name = f_name
        # ...

        # ...
        body        = []
        local_vars  = []
        global_vars = []
        imports     = [Import('mpi')]
        hide        = False
        kind        = 'procedure'
        cls_name    = '__UNDEFINED__'
        # ...

        # TODO add comm_parent as (optional) argument

        # ... args
        c_name = 'MPI_Tensor'
        alias  = None
        c_dtype = DataTypeFactory(c_name, ("_name"))

        this = Variable(c_dtype(), 'self')

        arg_x = Variable('double', 'arg_x', rank=2)
        args = [this, arg_x]
        # ...

        # ...
        ierr    = MPI_ERROR
        istatus = MPI_STATUS
        local_vars  += [ierr, istatus]
        # ...

        # ...
        results = []
        # ...

        # ... needed attributs
        comm          = Variable('int', DottedName('self', 'comm'), \
                                 cls_base=MPI_comm())

        type_line     = Variable('int', DottedName('self', 'line'))
        type_column   = Variable('int', DottedName('self', 'column'))

        starts        = IndexedVariable(DottedName('self', 'starts'), \
                                        dtype=NativeInteger())
        ends          = IndexedVariable(DottedName('self', 'ends'), \
                                        dtype=NativeInteger())
        neighbor      = IndexedVariable(DottedName('self', 'neighbor'), \
                                        dtype=NativeInteger())

        sx = starts[0] ; sy = starts[1]
        ex = ends[0]   ; ey = ends[1]

        north = 0 ; east = 1 ; south = 2 ; west = 3
        # ...

        # ... local variable
        tag_value = int(str(abs(hash(this)))[-6:])
        tag_name = 'tag_{0}'.format(str(tag_value))
        tag = Variable('int', tag_name)

        local_vars += [tag]
        body       += [Assign(tag, tag_value)]
        # ...

        # ... # TODO loop over variables
#        var = variables[0]
        var = arg_x
        var = IndexedVariable(var.name, dtype=type_line, shape=var.shape)

        # ...
        body.append(Comment('... MPI_Tensor: Send to neighbour N and receive from neighbour S'))
        # ...

        # Send to neighbour N and receive from neighbour S
        rhs = MPI_comm_sendrecv(var[sx, sy],   neighbor[north], tag, \
                                var[ex+1, sy], neighbor[south], tag, comm)
        stmt = MPI_Assign(ierr, rhs, strict=False)
        body.append(stmt)

        # ...
        body.append(Comment('...'))
        body.append(Comment('... MPI_Tensor: Send to neighbour S and receive from neighbour N'))
        # ...

        # Send to neighbour S and receive from neighbour N
        rhs = MPI_comm_sendrecv(var[ex, sy],   neighbor[south], tag, \
                                var[sx-1, sy], neighbor[north], tag, comm)
        stmt = MPI_Assign(ierr, rhs, strict=False)
        body.append(stmt)
        # ...

        # ...
        var = IndexedVariable(var.name, dtype=type_column, shape=var.shape)

        # ...
        body.append(Comment('...'))
        body.append(Comment('... MPI_Tensor: Send to neighbour W  and receive from neighbour E'))
        # ...

        # Send to neighbour W  and receive from neighbour E
        rhs = MPI_comm_sendrecv(var[sx, sy],   neighbor[west], tag, \
                                var[sx, ey+1], neighbor[east], tag, comm)
        stmt = MPI_Assign(ierr, rhs, strict=False)
        body.append(stmt)

        # ...
        body.append(Comment('...'))
        body.append(Comment('... MPI_Tensor: Send to neighbour E  and receive from neighbour W'))
        # ...

        # Send to neighbour E  and receive from neighbour W
        rhs = MPI_comm_sendrecv(var[sx, ey],   neighbor[east], tag, \
                                var[sx, sy-1], neighbor[west], tag, comm)
        stmt = MPI_Assign(ierr, rhs, strict=False)
        body.append(stmt)
        # ...

        # ...
        body.append(Comment('...'))
        body.append(EmptyLine())
        # ...

        return FunctionDef.__new__(cls, f_name, args, results, \
                                   body, local_vars, global_vars, \
                                   hide=hide, \
                                   kind=kind, \
                                   cls_name=cls_name, \
                                   imports=imports)

    @property
    def name(self):
        return self._name

    def _sympystr(self, printer):
        sstr = printer.doprint

        name    = 'MPI_Tensor_{0}'.format(sstr(self.name))
        args    = ', '.join(sstr(i) for i in self.arguments)
        results = ', '.join(sstr(i) for i in self.results)
        return '{0} := {1}({2})'.format(results, name, args)


class MPI_Tensor_free(FunctionDef):
    """
    Represents a Tensor free procedure.

    Examples

    >>> from pyccel.parallel.mpi import MPI_Tensor_free
    >>> MPI_Tensor_free()
    """
    def __new__(cls, attributs):
        """
        Represents a call to free for MPI tensor.
        """
        # ...
        f_name = '__del__'

        cls._name = f_name
        # ...

        # ...
        body        = []
        local_vars  = []
        global_vars = []
        imports     = [Import('mpi')]
        hide        = False
        kind        = 'procedure'
        cls_name    = '__UNDEFINED__'
        # ...

        # TODO add comm_parent as (optional) argument

        # ... args
        c_name = 'MPI_Tensor'
        alias  = None
        c_dtype = DataTypeFactory(c_name, ("_name"))

        this = Variable(c_dtype(), 'self')

        args = [this]
        # ...

        # ...
        ierr    = MPI_ERROR
        istatus = MPI_STATUS
        local_vars  += [ierr, istatus]
        # ...

        # ...
        results = []
        # ...

        # ... constructs the __del__ method if not provided
        _args = []
        for a in attributs:
            if isinstance(a, Variable):
                if a.allocatable:
                    _args.append(a)

        name_me = lambda a: DottedName(str(this), str(a.name))
        _args = [Variable(a.dtype, name_me(a)) for a in _args]
        body += [Del(a) for a in _args]
        # ...

        # ...
        body.append(Comment('...'))
        body.append(EmptyLine())
        # ...

        # ...
        comm = Variable('int', DottedName(this.name, 'comm'), \
                        cls_base=MPI_comm())

        body += [MPI_comm_free(comm)]
        # ...

        return FunctionDef.__new__(cls, f_name, args, results, \
                                   body, local_vars, global_vars, \
                                   hide=hide, \
                                   kind=kind, \
                                   cls_name=cls_name, \
                                   imports=imports)

    @property
    def name(self):
        return self._name

    def _sympystr(self, printer):
        sstr = printer.doprint

        name    = 'MPI_Tensor_{0}'.format(sstr(self.name))
        args    = ', '.join(sstr(i) for i in self.arguments)
        results = ', '.join(sstr(i) for i in self.results)
        return '{0} := {1}({2})'.format(results, name, args)

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
        ex = ends[0]   ; ey = ends[1]

        type_line   = tensor.types_bnd['line']
        type_column = tensor.types_bnd['column']

        north = 0 ; east = 1 ; south = 2 ; west = 3
        neighbor = tensor.neighbor

        comm = tensor.comm
        tag  = tensor.tag
        # ...

        # ... # TODO loop over variables
        var = variables[0]
        var = IndexedVariable(var.name, dtype=type_line, shape=var.shape)

        # ...
        body.append(Comment('... MPI_Tensor: Send to neighbour N and receive from neighbour S'))
        # ...

        # Send to neighbour N and receive from neighbour S
        rhs = MPI_comm_sendrecv(var[sx, sy],   neighbor[north], tag, \
                                var[ex+1, sy], neighbor[south], tag, comm)
        stmt = MPI_Assign(ierr, rhs, strict=False)
        body.append(stmt)

        # ...
        body.append(Comment('...'))
        body.append(Comment('... MPI_Tensor: Send to neighbour S and receive from neighbour N'))
        # ...

        # Send to neighbour S and receive from neighbour N
        rhs = MPI_comm_sendrecv(var[ex, sy],   neighbor[south], tag, \
                                var[sx-1, sy], neighbor[north], tag, comm)
        stmt = MPI_Assign(ierr, rhs, strict=False)
        body.append(stmt)
        # ...

        # ... # TODO loop over variables
        var = variables[0]
        var = IndexedVariable(var.name, dtype=type_column, shape=var.shape)

        # ...
        body.append(Comment('...'))
        body.append(Comment('... MPI_Tensor: Send to neighbour W  and receive from neighbour E'))
        # ...

        # Send to neighbour W  and receive from neighbour E
        rhs = MPI_comm_sendrecv(var[sx, sy],   neighbor[west], tag, \
                                var[sx, ey+1], neighbor[east], tag, comm)
        stmt = MPI_Assign(ierr, rhs, strict=False)
        body.append(stmt)

        # ...
        body.append(Comment('...'))
        body.append(Comment('... MPI_Tensor: Send to neighbour E  and receive from neighbour W'))
        # ...

        # Send to neighbour E  and receive from neighbour W
        rhs = MPI_comm_sendrecv(var[sx, ey],   neighbor[east], tag, \
                                var[sx, sy-1], neighbor[west], tag, comm)
        stmt = MPI_Assign(ierr, rhs, strict=False)
        body.append(stmt)
        # ...

        # ...
        body.append(Comment('...'))
        body.append(EmptyLine())
        # ...

        return super(MPI_TensorCommunication, cls).__new__(cls, local_vars, body)

    @property
    def tensor(self):
        return self._tensor

class MPI_CommunicationAction(MPI_Communication, Block):
    """MPI communication action over a MPI_Tensor object."""
    is_integer = True

    def __new__(cls, tensor, variables, action, options):
        if not isinstance(tensor, MPI_Tensor):
            raise TypeError('Expecting MPI_Tensor')

        if not iterable(variables):
            raise TypeError('Expecting an iterable of variables')

        if not isinstance(action, str):
            raise TypeError('Expecting a string')

        if not isinstance(options, list):
            raise TypeError('Expecting a list')

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

        comm = tensor.comm
        if action == 'allreduce':
            op = options[0]
            if op is None:
                raise ValueError('Expecting an operation')

            for var in variables:
                rhs = MPI_comm_allreduce(var, var, op, comm)
                stmt = MPI_Assign(ierr, rhs, strict=False)
                body.append(stmt)

        return super(MPI_CommunicationAction, cls).__new__(cls, local_vars, body)

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
    if isinstance(stmt, (list, tuple, Tuple)):
        return [mpify(i, **options) for i in stmt]
    if isinstance(stmt, MPI):
        return stmt
    if isinstance(stmt, Tensor):
        options['label'] = stmt.name
        return stmt
#        return MPI_Tensor(stmt, **options)
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
            master    = mpify(stmt.master, **options)
            action    = stmt.action

            # master can be a variable
            if isinstance(master, Variable):
                dtype = master.dtype
                cls   = eval(dtype.name)()
                methods = {}
                for i in cls.methods:
                    methods[str(i.name)] = i
                method = methods['communicate']
                args = [master] + list(variables)
                return MethodCall(method, args)
            else:
                raise NotImplementedError('Only available for Variable instance')

    return stmt

class MPI_Init(FunctionDef, MPI):
    """
    Call to MPI_init.

    Example
    >>> from pyccel.parallel.mpi import MPI_Init
    >>> MPI_Init()
    ierr := mpi_init()
    """
    def __new__(cls):
        body        = []
        local_vars  = []
        global_vars = []
        hide        = True
        kind        = 'procedure'

        args        = []
        datatype    = 'int'
        allocatable = False
        shape       = None
        rank        = 0

        var_name = 'result_%d' % abs(hash(datatype))

        f_name = 'mpi_init'

        var = Variable(datatype, var_name)
        results = [var]

        return FunctionDef.__new__(cls, f_name, args, results, \
                                   body, local_vars, global_vars, \
                                   hide=hide, kind=kind)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return 'ierr := {}()'.format(sstr(self.name))

class MPI_Finalize(FunctionDef, MPI):
    """
    Call to MPI_finalize.

    Example
    >>> from pyccel.parallel.mpi import MPI_Finalize
    >>> MPI_Finalize()
    ierr := mpi_finalize()
    """
    def __new__(cls):
        body        = []
        local_vars  = []
        global_vars = []
        hide        = True
        kind        = 'procedure'

        args        = []
        datatype    = 'int'
        allocatable = False
        shape       = None
        rank        = 0

        var_name = 'result_%d' % abs(hash(datatype))

        f_name = 'mpi_finalize'

        var = Variable(datatype, var_name)
        results = [var]

        return FunctionDef.__new__(cls, f_name, args, results, \
                                   body, local_vars, global_vars, \
                                   hide=hide, kind=kind)

    def _sympystr(self, printer):
        sstr = printer.doprint
        return 'ierr := {}()'.format(sstr(self.name))

##########################################################

MPI_ERROR   = Variable('int', 'i_mpi_error')
MPI_STATUS  = Variable(MPI_status_type(), 'i_mpi_status')

MPI_COMM_WORLD  = MPI_comm_world()
MPI_STATUS_SIZE = MPI_status_size()
MPI_PROC_NULL   = MPI_proc_null()

# ...
def mpi_definitions_OLD():
    """Adds MPI functions and constants to the namespace

    Returns

    namespace: dict
        dictorionary containing all declared variables/functions/classes.

    declarations: dict
        dictorionary containing all declarations.

    cls_constructs: dict
        dictionary of datatypes of classes using DatatypeFactory
    """
    namespace      = {}
    declarations   = {}
    cls_constructs = {}

    # TODO implement MPI_Init and Finalize classes like in clapp/plaf/matrix.py
    # ...
    namespace['mpi_comm_world']  = MPI_COMM_WORLD
    namespace['mpi_status_size'] = MPI_STATUS_SIZE
    namespace['mpi_proc_null']   = MPI_PROC_NULL

#    # TODO functions
#    namespace['mpi_init'] = MPI_Init()
#    namespace['mpi_finalize'] = MPI_Finalize()
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

    # ...
    classes = ['MPI_Tensor']
    for i in classes:
        name = 'pcl_t_{0}'.format(i.lower())
        cls_constructs[name] = DataTypeFactory(i, ("_name"))

        namespace[name] = eval(i)()
    # ...

    # ...
    stmts = []
    # ...

    return namespace, declarations, cls_constructs, stmts
# ...

# ...
def mpi_definitions():
    """Adds MPI functions and constants to the namespace

    Returns

    namespace: dict
        dictorionary containing all declared variables/functions/classes.

    declarations: dict
        dictorionary containing all declarations.

    cls_constructs: dict
        dictionary of datatypes of classes using DatatypeFactory
    """
    namespace      = {}
    declarations   = {}
    cls_constructs = {}
    classes        = {}
    stmts          = []

    # TODO implement MPI_Init and Finalize classes like in clapp/plaf/matrix.py
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
    for i in [MPI_Init(), MPI_Finalize()]:
        namespace[str(i.name)] = i
    # ...

    # ...
    c_name = 'MPI_Tensor'
    c_dtype = DataTypeFactory(c_name, ("_name"))
    cls_constructs[str(c_dtype.name)] = c_dtype()
    classes['MPI_Tensor'] = MPI_Tensor()

    stmts += [MPI_Tensor()]
    # ...



#    # ...
#    body        = []
#    local_vars  = []
#    global_vars = []
#    hide        = True
#    kind        = 'procedure'
#    # ...
#
#    # ...
#    args        = []
#    datatype    = 'int'
#    allocatable = False
#    shape       = None
#    rank        = 0
#
#    var_name = 'result_%d' % abs(hash(datatype))
#
#    for f_name in ['mpi_init', 'mpi_finalize']:
#        var = Variable(datatype, var_name)
#        results = [var]
#
#        stmt = FunctionDef(f_name, args, results, \
#                           body, local_vars, global_vars, \
#                           hide=hide, kind=kind)
#
#        namespace[f_name] = stmt
#    # ...


    return namespace, declarations, cls_constructs, classes, stmts
# ...
