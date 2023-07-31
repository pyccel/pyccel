# pylint: disable=missing-function-docstring, missing-module-docstring
from mpi4py import MPI


#==============================================================================

#
# comm.Sendrecv( sendbuf, dest, sendtag, recvbuf, source, recvtag, status )
#

def np_sendrecv(sendbuf : 'int[:]', dest : int):

    comm = MPI.COMM_WORLD
    recvbuf[:] = 0

    comm.Sendrecv( sendbuf, dest, sendtag, recvbuf, source, recvtag )

