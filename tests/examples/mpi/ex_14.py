# coding: utf-8

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

#dims    = array((4, 2), int)
#periods = array((False, True), bool)
period = True
period = False

ierr = mpi_finalize()
