# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

r_x = range(0, 8)
r_y = range(0, 8)


#Grid spacing
ntx = 8
nty = 8
hx = 1.0/(ntx+1)
hy = 1.0/(nty+1)

#Equation Coefficients
c0 = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy)
c1 = 1.0/(hx*hx)
c2 = 1.0/(hy*hy)

mesh = tensor(r_x, r_y)

u       = zeros(mesh, double)
u_new   = zeros(mesh, double)
u_exact = zeros(mesh, double)
f       = zeros(mesh, double)

#Initialization
for i,j in mesh:
    x = i*hx
    y = j*hy

    f[i, j] = 2*(x*x-x+y*y-y)
    u_exact[i, j] = x*y*(x-1)*(y-1)


n_iterations = 1
for it in range(0, n_iterations):
    u = u_new

    #Communication
    sync(mesh) u

    #Computation of u at the n+1 iteration
    for i,j in mesh:
        u_new[i, j] = c0 * (c1*(u[i+1, j] + u[i-1, j]) + c2*(u[i, j+1] + u[i, j-1]) - f[i, j])

del mesh
ierr = mpi_finalize()
