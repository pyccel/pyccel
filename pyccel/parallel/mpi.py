# coding: utf-8

from sympy.core.basic   import Basic
from sympy.core.symbol  import Symbol
from sympy.core.numbers import Integer

from pyccel.types.ast import Variable
from pyccel.parallel.communicator import UniversalCommunicator

class MPI(Basic):
    pass

class MPI_world_comm(UniversalCommunicator, MPI):
    """Represents the world comm in mpi."""
    is_integer     = True

    def _sympystr(self, printer):
        sstr = printer.doprint
        return 'mpi_comm_world'

MPI_WORLD_COMM = MPI_world_comm()
