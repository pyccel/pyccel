# coding: utf-8

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

dims    = array((4, 2), int)
periods = array((False, True), bool)

reorder = False
ierr = comm.cart_create(dims, periods, reorder, comm_2d)

#Destruction of the communicators
ierr = comm_2d.free()

ierr = mpi_finalize()
