# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

if rank == 0:
    partner = 1

if rank == 1:
    partner = 0

msg = rank + 1000
tag = 1234
ierr = comm.sendrecv_replace (msg, partner, tag, partner, tag)

print(('I, process ', rank, ', I received', msg, ' from process ', partner))

ierr = mpi_finalize()

# TODO: - example barrier
