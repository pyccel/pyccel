# coding: utf-8

# usage:
#   > pyccel poisson_mpi_v1.py  --include='$INCLUDE_DIR' --libdir='$LIB_DIR' --libs=poisson --compiler=mpif90 --no-modules

from pyccel.stdlib.parallel.mpi import mpi_init
from pyccel.stdlib.parallel.mpi import mpi_finalize
from pyccel.stdlib.parallel.mpi import mpi_allreduce
from pyccel.stdlib.parallel.mpi import MPI_INTEGER
from pyccel.stdlib.parallel.mpi import MPI_DOUBLE
from pyccel.stdlib.parallel.mpi import MPI_SUM

from pyccel.stdlib.parallel.mpi import Cart

from pyccelext.math.quadratures  import legendre
from pyccelext.math.external.bsp import spl_make_open_knots
from pyccelext.math.external.bsp import spl_compute_spans
from pyccelext.math.external.bsp import spl_compute_origins_element
from pyccelext.math.external.bsp import spl_construct_grid_from_knots
from pyccelext.math.external.bsp import spl_construct_quadrature_grid
from pyccelext.math.external.bsp import spl_eval_on_grid_splines_ders

ierr = -1

mpi_init(ierr)

# ...
p1 = 1
p2 = 1

n_elements_1 = 8
n_elements_2 = 8

n_elements_1 = n_elements_1 - p1
n_elements_2 = n_elements_2 - p2

# number of derivatives
d1 = 1
d2 = 1

n1 = p1 + n_elements_1
n2 = p2 + n_elements_2

k1 = p1 + 1
k2 = p2 + 1

verbose = False
#verbose = True
# ...

# ...
npts    = zeros(2, int)
degrees = ones(2, int)
pads    = ones(2, int)
periods = zeros(2, bool)
reorder = False
# ...

# ...
npts[0] = n1
npts[1] = n2

degrees[0] = p1
degrees[1] = p2

pads[0] = p1
pads[1] = p2
# ...

mesh = Cart(npts, pads, periods, reorder)

# ...
sx = mesh.starts[0]
ex = mesh.ends[0]

sy = mesh.starts[1]
ey = mesh.ends[1]
# ...

# ...
[u1,w1] = legendre(p1)
# ...

# ...
[u2,w2] = legendre(p2)
# ...

# ...
m1 = n1 + p1 + 1
m2 = n2 + p2 + 1

knots1 = zeros(m1, double)
knots2 = zeros(m2, double)

# call to spl
knots1 = spl_make_open_knots (n1, p1)

# call to spl
knots2 = spl_make_open_knots (n2, p2)
# ...

# ... TODO fix args of zeros
m1 = n_elements_1+1
m2 = n_elements_2+1

grid_1 = zeros(m1, double)
grid_2 = zeros(m2, double)

# call to spl
grid_1 = spl_construct_grid_from_knots(p1, n1, n_elements_1, knots1)

# call to spl
grid_2 = spl_construct_grid_from_knots(p2, n2, n_elements_2, knots2)
# ...

# ... construct the quadrature points grid
points_1  = zeros((k1, n_elements_1), double)
points_2  = zeros((k2, n_elements_2), double)
weights_1 = zeros((k1, n_elements_1), double)
weights_2 = zeros((k2, n_elements_2), double)

# call to spl
[points_1, weights_1] = spl_construct_quadrature_grid(u1, w1, grid_1)

# call to spl
[points_2, weights_2] = spl_construct_quadrature_grid(u2, w2, grid_2)
# ...

# ...
basis_1  = zeros((p1+1, d1+1, k1, n_elements_1), double)
basis_2  = zeros((p2+1, d2+1, k2, n_elements_2), double)

# call to spl
basis_1 = spl_eval_on_grid_splines_ders(n1, p1, d1, knots1, points_1)

# call to spl
basis_2 = spl_eval_on_grid_splines_ders(n2, p2, d2, knots2, points_2)
# ...

# ...
spans_1 = zeros(n_elements_1, int)
spans_2 = zeros(n_elements_2, int)

spans_1 = spl_compute_spans(p1, n1, knots1)
spans_2 = spl_compute_spans(p2, n2, knots2)
# ...

# ...
origins_1 = zeros(n1, int)
origins_2 = zeros(n2, int)

origins_1 = spl_compute_origins_element(p1, n1, knots1)
origins_2 = spl_compute_origins_element(p2, n2, knots2)
# ...

# ...
start_1 = 0
end_1   = n1-1
pad_1   = p1

start_2 = 0
end_2   = n2-1
pad_2   = p2
# ...

# ...
mass      = stencil((start_1, start_2), (end_1, end_2), (pad_1, pad_2))
stiffness = stencil((start_1, start_2), (end_1, end_2), (pad_1, pad_2))
rhs       = vector((start_1-pad_1, start_2-pad_2), (end_1+pad_1, end_2+pad_2))
# ...

# ...
print('(', sx, ',', ex, ')  (', sy, ',', ey, ')')
print(origins_1(sx), origins_2(sy))
# ...

# ...
u       = vector((sx-1,sy-1), (ex+1, ey+1))
u_new   = vector((sx-1,sy-1), (ex+1, ey+1))
u_exact = vector((sx-1,sy-1), (ex+1, ey+1))
f       = vector((sx-1,sy-1), (ex+1, ey+1))
# ...

# ...
# Initialization
x = 0.0
y = 0.0
#for i,j in mesh.extended_indices:
#    print(i,j)
# ...

del mesh

mpi_finalize(ierr)
