from mpi4py import MPI

from pyccel.decorators import types

#==============================================================================

#
# comm.Sendrecv( sendbuf, dest, sendtag, recvbuf, source, recvtag, status )
#

@types( 'int[:]', int, int, 'int[:]', int, int )
def np_sendrecv( sendbuf, dest, sendtag, recvbuf, source, recvtag ):

    comm = MPI.COMM_WORLD
    ierr = -1
    recvbuf[:] = 0

    comm.Sendrecv( sendbuf, dest, sendtag, recvbuf, source, recvtag )

