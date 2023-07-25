# pylint: disable=missing-function-docstring, missing-module-docstring
from mpi4py import MPI

if __name__ == '__main__':
    rank = -1
    #we must initialize rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        partner = 1

    if rank == 1:
        partner = 0

    msg = rank + 1000
    val = -1
    tag = 1234

    val = comm.sendrecv(msg, 1, tag, partner, tag)
    print('I, process ', rank, ', I received', val, ' from process ', partner)
