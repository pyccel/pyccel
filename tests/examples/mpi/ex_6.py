# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

root = 1
if rank == root:
    value = rank + 1000

ierr = comm.bcast (value, root)

print(('I, process ', rank, ', received ', value, ' from process ', root))

ierr = mpi_finalize()
