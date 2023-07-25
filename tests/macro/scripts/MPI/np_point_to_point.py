# pylint: disable=missing-function-docstring, missing-module-docstring
from mpi4py import MPI
from numpy import zeros
from numpy import ones


if __name__ == '__main__':
    rank = -1
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # passing MPI datatypes explicitly
    if rank == 0:
        data = ones(5, 'int')
        comm.Send([data, MPI.INT], dest=1, tag=77)
    elif rank == 1:
        data = zeros(5, 'int')
        comm.Recv([data, MPI.INT], source=0, tag=77)
        print(data)

    # automatic MPI datatype discovery
    if rank == 0:
        data_ = ones(5, 'double')
        comm.Send(data, dest=1, tag=13)
    elif rank == 1:
        data_ = zeros(5, 'double')
        comm.Recv(data, source=0, tag=13)
        print(data)
