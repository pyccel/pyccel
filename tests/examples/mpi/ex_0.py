# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

print(('I process ', rank, ', among ', size, ' processes'))

ierr = mpi_finalize()
