# coding: utf-8

from pyccel.mpi import *

ierr = mpi_init()

comm = mpi_comm_world
size = comm.size
rank = comm.rank

ntx = 128
nty = 128
r_x = range(0, ntx)
r_y = range(0, nty)

#Grid spacing
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
x = 0.0
y = 0.0
for i,j in mesh:
    x = i*hx
    y = j*hy

    f[i, j] = 2.0*(x*x-x+y*y-y)
    u_exact[i, j] = x*y*(x-1.0)*(y-1.0)

#Linear solver tolerance
tol = 1.0e-10

n_iterations = 100000
for it in range(0, n_iterations):
    u = u_new

    #Communication
    sync(mesh) u

    #Computation of u at the n+1 iteration
    for i,j in mesh:
        u_new[i, j] = c0 * (c1*(u[i+1, j] + u[i-1, j]) + c2*(u[i, j+1] + u[i, j-1]) - f[i, j])

    #Computation of the global error
    u_error = 0.0
    for i,j in mesh:
        u_error += abs(u[i,j]-u_new[i,j])
    local_error = u_error/(ntx*nty)

    #Reduction
    global_error = local_error
    sync(mesh, 'allreduce', '+') global_error

    if global_error < tol:
        if rank == 0:
            print ("convergence after ", it, " iterations")
            print ("local  error = ", local_error)
            print ("global error = ", global_error)
        break

#Computation of the global error
u_error = 0.0
for i,j in mesh:
    u_error += abs(u[i,j]-u_exact[i,j])
u_error = u_error/(ntx*nty)

#Reduction
sync(mesh, 'allreduce', '+') u_error

if rank == 0:
    print ("error(u_h - u_exact) = ", u_error)

del mesh
ierr = mpi_finalize()
