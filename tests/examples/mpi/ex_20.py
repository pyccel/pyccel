# coding: utf-8

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

u = range(0, 4)
v = range(0, 4)

ierr = mpi_finalize()
