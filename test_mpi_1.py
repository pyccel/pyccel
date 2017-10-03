# coding: utf-8

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

n = 4
x = zeros(n, double)

if rank == 0:
    x = 1.0

source = 0
dest   = 1
tag = 1234
if rank == source:
    ierr = comm.send(x, dest, tag)
    print("processor ", rank, " sent ", x)

if rank == dest:
    ierr = comm.recv(x, source, tag)
    print("processor ", rank, " got  ", x)

ierr = mpi_finalize()



#   integer numtasks, rank, next, prev, buf(2), tag1, tag2, ierr
#   integer reqs(4)   ! required variable for non-blocking calls
#   integer stats(MPI_STATUS_SIZE,4)   ! required variable for WAITALL routine
#
#
#   ! wait for all non-blocking operations to complete
#   call MPI_WAITALL(4, reqs, stats, ierr);

