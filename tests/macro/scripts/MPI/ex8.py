from mppy4py import MPI

# we need to declare these variables somehow,
# since we are calling mpi subroutines
size = -1
rank = -1



comm = mpi_comm_world
mpi_comm_size(comm, size, ierr)
mpi_comm_rank(comm, rank, ierr)

master    = 1
nb_values = 8

block_length = nb_values / size

# ...
values = zeros(block_length, 'int')
for i in range(0, block_length):
    values[i] = 1000 + rank*nb_values + i

print('I, process ', rank, 'sent my values array : ', values)
# ...

# ...
data = zeros(nb_values, 'int')

mpi_gather (values, block_length, MPI_INTEGER,
            data,   block_length, MPI_INTEGER,
            master, comm, ierr)
# ...

if rank == master:
    print('I, process ', rank, ', received ', data, ' of process ', master)

