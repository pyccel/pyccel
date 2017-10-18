# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

ntx = 8
nty = 8
r_x = range(0, ntx)
r_y = range(0, nty)

#Grid spacing
hx = 1.0/(ntx+1)
hy = 1.0/(nty+1)


npts    = (32,32)
periods = (False,False)
reorder = False
pads    = (1,1)

mesh = MPI_Tensor(npts, periods, reorder, pads)

starts = mesh.starts
ends   = mesh.ends

print ("(", starts[0], ", ", ends[0], ")   (", starts[1], ", ", ends[1], ')')

u       = zeros(mesh, double)
u_new   = zeros(mesh, double)
u_exact = zeros(mesh, double)
f       = zeros(mesh, double)

#Initialization
x = 0.0
y = 0.0
for i,j in mesh:
    x = i*hx
    y = j*hy

    f[i, j] = 2.0*(x*x-x+y*y-y)
    u_exact[i, j] = x*y*(x-1.0)*(y-1.0)

#Linear solver tolerance
tol = 1.0e-10

n_iterations = 1
for it in range(0, n_iterations):
    u = u_new

    #Communication
    sync(mesh) u

del mesh
ierr = mpi_finalize()
