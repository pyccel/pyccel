# pylint: disable=missing-function-docstring, missing-module-docstring
from mpi4py import MPI
from numpy import ones
from numpy import zeros

if __name__ == '__main__':
    rank = -1
    #we must initialize rank
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    root = 0

    if rank == 0:
        value = ones(50,'int')
    else:
        value = zeros(50,'int')

    sum_value = zeros(50,'int')
    comm.Allreduce (value, sum_value, MPI.SUM)

    if rank == 0:
        print('I, process ', root,', have the global sum value ', sum_value)

