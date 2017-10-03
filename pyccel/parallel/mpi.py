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
    pass

class MPI_Statement(Basic):
    pass

class MPI_Assign(Assign, MPI_Statement):
    pass

class MPI_Declare(Declare, MPI_Statement):
    pass

class MPI_comm_world(UniversalCommunicator, MPI):
    """Represents the world comm in mpi."""
    is_integer     = True

    def _sympystr(self, printer):
        sstr = printer.doprint
        return 'mpi_comm_world'

class MPI_comm_size(MPI):
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_size, cls).__new__(cls, *args, **options)

    @property
    def comm(self):
        return self.args[0]

class MPI_comm_rank(MPI):
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_rank, cls).__new__(cls, *args, **options)

    @property
    def comm(self):
        return self.args[0]

class MPI_comm_recv(MPI):
    is_integer = True

    def __new__(cls, *args, **options):
        return super(MPI_comm_recv, cls).__new__(cls, *args, **options)

    @property
    def comm(self):
        return self.args[0]

class MPI_comm_send(MPI):
#    MPI_SEND(data, size, datatype, dest, tag, comm)
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
    def size(self):
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
        dtype = self.data.dtype
        if isinstance(dtype, NativeInteger):
            dtype = 'MPI_INT'
        elif isinstance(dtype, NativeFloat):
            dtype = 'MPI_REAL'
        elif isinstance(dtype, NativeDouble):
            dtype = 'MPI_DOUBLE'
        else:
            raise TypeError("Uncovered datatype.")
        return dtype

class MPI_status_type(DataType):
    pass


MPI_ERROR  = Variable('int', 'i_mpi_error')
MPI_STATUS = Variable(MPI_status_type(), 'i_mpi_status')
MPI_COMM_WORLD = MPI_comm_world()


