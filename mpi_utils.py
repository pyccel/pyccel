# coding: utf-8

from pyccel.stdlib.parallel.mpi import mpi_init
from pyccel.stdlib.parallel.mpi import mpi_finalize

from pyccel.stdlib.parallel.mpi_new import Cart

ierr = -1

mpi_init(ierr)

p = Cart()

del p

mpi_finalize(ierr)
