from mpi4py import MPI
from numpy  import empty

from pyccel.decorators import types

# TODO: avoid declaration of integer variables 'ierr'
# TODO: allow passing MPI communicator to functions
# TODO: understand that 'recvbuf' has intent(inout)

#==============================================================================

@types( 'int[:]', 'int[:]' )
def np_allreduce( sendbuf, recvbuf ):

    comm = MPI.COMM_WORLD
    ierr = -1
    recvbuf[:] = 0

    comm.Allreduce( sendbuf, recvbuf, MPI.SUM )

