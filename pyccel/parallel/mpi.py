# coding: utf-8

from sympy.core.symbol  import Symbol
from sympy.core.numbers import Integer

from pyccel.types.ast             import Variable
from pyccel.types.ast             import Assign

from pyccel.parallel.basic        import Basic
from pyccel.parallel.communicator import UniversalCommunicator

class MPI(Basic):
    pass

class MPI_Statement(Basic):
    pass

class MPI_Assign(Assign, MPI_Statement):
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


MPI_ERROR = Variable('int', 'i_mpi_error')
MPI_COMM_WORLD = MPI_comm_world()
