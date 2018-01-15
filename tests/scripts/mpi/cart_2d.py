# coding: utf-8

from pyccel.stdlib.parallel.mpi import mpi_init
from pyccel.stdlib.parallel.mpi import mpi_finalize

from pyccel.stdlib.parallel.mpi import mpi_allreduce
from pyccel.stdlib.parallel.mpi import MPI_INTEGER
from pyccel.stdlib.parallel.mpi import MPI_DOUBLE
from pyccel.stdlib.parallel.mpi import MPI_SUM

from pyccel.stdlib.parallel.mpi import Cart

ierr = -1

mpi_init(ierr)

# ...
npts    = zeros(2, int)
pads    = ones(2, int)
periods = zeros(2, bool)
reorder = False
# ...

# ...
npts[0] = 32
npts[1] = 32
# ...

mesh = Cart(npts, pads, periods, reorder)

# ...
sx = mesh.starts[0]
ex = mesh.ends[0]

sy = mesh.starts[1]
ey = mesh.ends[1]
# ...

# ...
#print('(', sx, ',', ex, ')  (', sy, ',', ey, ')')
# ...

# ...
u       = vector((sx-1,sy-1), (ex+1, ey+1))
u_new   = vector((sx-1,sy-1), (ex+1, ey+1))
u_exact = vector((sx-1,sy-1), (ex+1, ey+1))
f       = vector((sx-1,sy-1), (ex+1, ey+1))
# ...

# ...
ntx = npts[0]
nty = npts[1]

# Grid spacing
hx = 1.0/(ntx+1)
hy = 1.0/(nty+1)

# Equation Coefficients
c0 = (0.5*hx*hx*hy*hy)/(hx*hx+hy*hy)
c1 = 1.0/(hx*hx)
c2 = 1.0/(hy*hy)
# ...

# Initialization
x = 0.0
y = 0.0
for i,j in mesh.extended_indices:
    x = i*hx
    y = j*hy

    f[i, j] = 2.0*(x*x-x+y*y-y)
    u_exact[i, j] = x*y*(x-1.0)*(y-1.0)
# ...

# Linear solver tolerance
tol = 1.0e-10

n_iterations = 80000
for it in range(0, n_iterations):
    u[sx:ex+1,sy:ey+1] = u_new[sx:ex+1,sy:ey+1]

    mesh.communicate(u)

    # ... Computation of u at the n+1 iteration
    for i,j in mesh.indices:
        u_new[i, j] = c0 * (c1*(u[i+1, j] + u[i-1, j]) + c2*(u[i, j+1] + u[i, j-1]) - f[i, j])
    # ...

    # ... Computation of the global error
    u_error = 0.0
    for i,j in mesh.indices:
        u_error += abs(u[i,j]-u_new[i,j])
    local_error = u_error/(ntx*nty)

    # Reduction
    ierr = -1
    global_error = 0.0
    mpi_allreduce (local_error, global_error, 1, MPI_DOUBLE, MPI_SUM, mesh.comm_cart, ierr)
    # ...

    # ...
    if (global_error < tol) or (it == n_iterations - 1):
        if mesh.rank == 0:
            print ("> convergence after ", it, " iterations")
            print ("  local  error = ", local_error)
            print ("  global error = ", global_error)
        break
    # ...

del mesh

mpi_finalize(ierr)
