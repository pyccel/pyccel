# pylint: disable=missing-function-docstring, missing-module-docstring
from mpi4py import MPI


# TODO: avoid declaration of integer variables 'ierr' and 'rank'
# TODO: allow access to process rank through property 'comm.rank'
# TODO: allow passing MPI communicator to functions
# TODO: understand that 'recvbuf' has intent(inout)

#==============================================================================

def np_allreduce(sendbuf : 'int[:]', recvbuf : 'int[:]'):

    comm = MPI.COMM_WORLD
    recvbuf[:] = 0

    comm.Allreduce( sendbuf, recvbuf, MPI.SUM )

# ...
def np_bcast(buf : 'int[:]', root : int):

    comm = MPI.COMM_WORLD
    rank = -1
    rank = comm.Get_rank()

    if rank != root:
        buf[:] = 0

    comm.Bcast( buf, root )

# ...
def np_gather(sendbuf : 'int[:]', recvbuf : 'int[:]', root : int):

    comm = MPI.COMM_WORLD
    rank = -1
    rank = comm.Get_rank()

    if rank == root:
        recvbuf[:] = 0

    comm.Gather( sendbuf, recvbuf, root )
