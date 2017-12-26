# coding: utf-8

from pyccel.stdlib.parallel.mpi import mpi_init
from pyccel.stdlib.parallel.mpi import mpi_finalize

from pyccel.stdlib.parallel.mpi_new import Cart

ierr = -1

mpi_init(ierr)

mesh = Cart()

#sx = mesh.starts[0]
#ex = mesh.ends[0]
#
#sy = mesh.starts[1]
#ey = mesh.ends[1]

#r_ext_x = range(sx-1, ex+1+1)
#r_ext_y = range(sy-1, ey+1+1)
#mesh_ext = tensor(r_ext_x, r_ext_y)

del mesh

mpi_finalize(ierr)
