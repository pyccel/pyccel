# coding: utf-8

ierr = mpi_init()

comm     = mpi_comm_world
nb_procs = comm.size
rank     = comm.rank

nb_values = 8

block_length = nb_values / nb_procs

values = zeros(block_length, double)
for i in range(0, block_length):
    values[i] = (rank + 1)* 1000 + i
print ('I, process ', rank, 'sent my values array : ', values)

data = zeros(nb_values, double)
ierr = comm.allgather(values, data)

print ('I, process ', rank, ', received ', data, ' of process ')

ierr = mpi_finalize()
