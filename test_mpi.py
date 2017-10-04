# coding: utf-8

ierr = mpi_init()

status_size = mpi_status_size

reqs  = zeros(4, int)
stats = zeros((status_size,4), int)

#ierr = mpi_waitall(reqs, stats)

ierr = mpi_finalize()
