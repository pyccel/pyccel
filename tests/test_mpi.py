# coding: utf-8

ierr = mpi_init()
comm = mpi_comm_world
print("mpi_comm = ", comm)

size, ierr = mpi_comm_size(comm)
print("mpi_size = ", size)

rank, ierr = mpi_comm_rank(comm)
print("mpi_rank = ", rank)

#abort, ierr = mpi_abort(comm)
#print("mpi_abort = ", abort)

ierr = mpi_finalize()
