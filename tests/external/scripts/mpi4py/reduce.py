# pylint: disable=missing-function-docstring, missing-module-docstring/
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
root = 0

if rank == 0:
    value = 1000
else:
    value = rank

sum_value = comm.reduce(value, MPI.SUM, root)

if rank == 0:
    print('I, process ', root,', have the global sum value ', sum_value)
