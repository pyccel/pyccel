# pylint: disable=missing-function-docstring, missing-module-docstring
from mpi4py import MPI
from numpy import zeros

# we need to declare these variables somehow,
# since we are calling mpi subroutines
size_ = -1
rank = -1
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_ = comm.Get_size()

master    = 1
nb_values = 8

block_length = nb_values // size_

# ...
values = zeros(block_length, 'int')
for i in range(0, block_length):
    values[i] = 1000 + rank*nb_values + i

print('I, process ', rank, 'sent my values array : ', values)
# ...

# ...
data = zeros(nb_values, 'int')

comm.Gather(values, data,master)
# ...

if rank == master:
    print('I, process ', rank, ', received ', data, ' of process ', master)

