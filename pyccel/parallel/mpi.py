# coding: utf-8

from sympy.core.symbol  import Symbol
from sympy.core.numbers import Integer


from pyccel.types.ast import Variable
from pyccel.types.ast import Assign, Declare
from pyccel.types.ast import NativeBool, NativeFloat, NativeComplex, NativeDouble, NativeInteger
from pyccel.types.ast import DataType
from pyccel.types.ast import DataTypeFactory

from pyccel.parallel.basic        import Basic
from pyccel.parallel.communicator import UniversalCommunicator

class MPI(Basic):
    """Base class for MPI."""
    pass

class MPI_Statement(Basic):
    """Base class for MPI statements."""
    pass

class MPI_Assign(Assign, MPI_Statement):
    """MPI statement that can be written as an assignment in pyccel."""
    pass

class MPI_Declare(Declare, MPI_Statement):
    """MPI declaration of a variable."""
    pass

class MPI_status_type(DataType):
    """Represents the datatype of MPI status."""
    pass

class MPI_INTEGER(DataType):
    _name = 'MPI_INTEGER'
    pass

class MPI_FLOAT(DataType):
    _name = 'MPI_FLOAT'
    pass


class MPI_DOUBLE(DataType):
    _name = 'MPI_DOUBLE'
    pass


def mpi_datatype(dtype):
    """Converts Pyccel datatypes into MPI datatypes."""
    if isinstance(dtype, NativeInteger):
        return 'MPI_INT'
    elif isinstance(dtype, NativeFloat):
        return 'MPI_REAL'
    elif isinstance(dtype, NativeDouble):
        return 'MPI_DOUBLE'
    else:
        raise TypeError("Uncovered datatype.")

class MPI_comm_world(UniversalCommunicator, MPI):
    """Represents the world comm in mpi."""
    is_integer     = True

    def _sympystr(self, printer):
        sstr = printer.doprint
        return 'mpi_comm_world'

class MPI_status_size(MPI):
    """Represents the status size in mpi."""
    is_integer     = True

    def _sympystr(self, printer):
        sstr = printer.doprint
        return 'mpi_status_size'

class MPI_comm_size(MPI):
    """Represents the size of a given communicator."""
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_size, cls).__new__(cls, *args, **options)

    @property
    def comm(self):
        return self.args[0]

class MPI_comm_rank(MPI):
    """Represents the process rank within a given communicator."""
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_rank, cls).__new__(cls, *args, **options)

    @property
    def comm(self):
        return self.args[0]

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
        shape = self.data.shape
        if isinstance(shape, (list, tuple)):
            n = 1
            for i in shape:
                n *= i
            return n
        else:
            return shape

    @property
    def datatype(self):
        return mpi_datatype(self.data.dtype)

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
        shape = self.data.shape
        if isinstance(shape, (list, tuple)):
            n = 1
            for i in shape:
                n *= i
            return n
        else:
            return shape

    @property
    def datatype(self):
        return mpi_datatype(self.data.dtype)

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
        shape = self.data.shape
        if isinstance(shape, (list, tuple)):
            n = 1
            for i in shape:
                n *= i
            return n
        else:
            return shape

    @property
    def datatype(self):
        return mpi_datatype(self.data.dtype)

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
        shape = self.data.shape
        if isinstance(shape, (list, tuple)):
            n = 1
            for i in shape:
                n *= i
            return n
        else:
            return shape

    @property
    def datatype(self):
        return mpi_datatype(self.data.dtype)

class MPI_comm_sendrecv(MPI):
    """
    Represents the MPI_sendrecv statement.
    MPI_sendrecv syntax is
    `MPI_SENDRECV (data, count, datatype, dest, tag, comm)`

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
        shape = self.data.shape
        if isinstance(shape, (list, tuple)):
            n = 1
            for i in shape:
                n *= i
            return n
        else:
            return shape

    @property
    def datatype(self):
        return mpi_datatype(self.data.dtype)

class MPI_Request(Variable, MPI):
    """
    Represents a MPI request variable.

    Examples

    >>> from pyccel.parallel.mpi import MPI_Request
    >>> req = MPI_Request('req')
    >>> req.dtype
    MPI_INTEGER()
    >>> req = MPI_Request('req', rank=1, shape=4)
    """

    def __new__(cls, *args, **options):
        dtype = MPI_INTEGER()
        args = [dtype] + list(args)
        return super(MPI_Request, cls).__new__(cls, *args, **options)


MPI_ERROR   = Variable('int', 'i_mpi_error')
MPI_STATUS  = Variable(MPI_status_type(), 'i_mpi_status')

MPI_COMM_WORLD  = MPI_comm_world()
MPI_STATUS_SIZE = MPI_status_size()
