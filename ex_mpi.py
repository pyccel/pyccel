# coding: utf-8

# TODO fix a bug when substructing f[i,j] in u_new and also when computing the
#      error

from pyccel.mpi import *

ierr = mpi_init()

npts    = (63,63)
periods = (False,False)
reorder = False
pads    = (1,1)

#Grid spacing
hx = 1.0/(npts[0]+1)
hy = 1.0/(npts[1]+1)

#Equation Coefficients
c0 = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy)
c1 = 1.0/(hx*hx)
c2 = 1.0/(hy*hy)

mesh = MPI_Tensor(npts, periods, reorder, pads)

rank = mesh.rank

starts = mesh.starts
ends   = mesh.ends

#print ("(", starts[0], ", ", ends[0], ")   (", starts[1], ", ", ends[1], ')')

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

n_iterations = 50000
for it in range(0, n_iterations):
    u = u_new

    #Communication
    sync(mesh) u

    #Computation of u at the n+1 iteration
    for i,j in mesh:
        u_new[i, j]  = c0 * (c1*(u[i+1, j] + u[i-1, j]) + c2*(u[i, j+1] + u[i, j-1]))
        u_new[i, j] -= c0 * f[i, j]

    #Computation of the global error
    u_error = 0.0
    for i,j in mesh:
        err  = u[i,j]
        err -= u_new[i,j]
        u_error += abs(err)
    local_error = u_error/(npts[0]*npts[1])

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
u_error = u_error/(npts[0]*npts[1])

#Reduction
sync(mesh, 'allreduce', '+') u_error

if rank == 0:
    print ("error(u_h - u_exact) = ", u_error)


del mesh
ierr = mpi_finalize()
