MPI
===

Enabling **MPI** is done in two steps:

- you need to have the following import:

  .. code-block:: python

    from pyccel.mpi import *

- you need to compile your file with a valid **mpi** compiler::

    pyccel --language="fortran" --compiler=mpif90 --filename=tests/examples/mpi/ex_0.py
    mpirun -n 2 tests/examples/mpi/ex_0

Let's start with a simple example (**tests/examples/mpi/ex_0.py**):

.. code-block:: python

  from pyccel.mpi import *

  ierr = mpi_init()

  comm = mpi_comm_world
  size = comm.size
  rank = comm.rank

  print ('I process ', rank, ', among ', size, ' processes')

  ierr = mpi_finalize()

we compile the file using::

    pyccel --language="fortran" --compiler=mpif90 --filename=tests/examples/mpi/ex_0.py

The generated *Fortran* code is

.. code-block:: fortran

  program main
  use MPI
  implicit none
  integer, dimension(MPI_STATUS_SIZE) :: i_mpi_status
  integer :: ierr
  integer :: comm
  integer :: rank
  integer :: i_mpi_error
  integer :: size

  !  
  call mpi_init(ierr)
  comm = MPI_comm_world
  call mpi_comm_size (comm, size, i_mpi_error)
  call mpi_comm_rank (comm, rank, i_mpi_error)
  print * ,'I process ',rank,', among ',size,' processes'
  call mpi_finalize(ierr)

  end

now let's run the executable::

  mpirun -n 4 tests/examples/mpi/ex_0

the result is::

  I process            1 , among            4  processes
  I process            2 , among            4  processes
  I process            3 , among            4  processes
  I process            0 , among            4  processes

Note that **comm** is considered as an object in our python file. A communicator has the following attributs:

- **size** : total number of processes within the communicator,

- **rank** : rank of the current process.

A communicator has also (many of) **MPI** procedures, defined as **methods**. The following example shows how to use the **send** and **recv** actions with respect to a given communicator.

(listing of **tests/examples/mpi/ex_1.py**)

.. code-block:: python

  # coding: utf-8

  from pyccel.mpi import *

  ierr = mpi_init()

  comm = mpi_comm_world
  size = comm.size
  rank = comm.rank

  n = 4
  x = zeros(n, double)
  y = zeros((3,2), double)

  if rank == 0:
      x = 1.0
      y = 1.0

  source = 0
  dest   = 1
  tagx = 1234
  if rank == source:
      ierr = comm.send(x, dest, tagx)
      print("processor ", rank, " sent ", x)

  if rank == dest:
      ierr = comm.recv(x, source, tagx)
      print("processor ", rank, " got  ", x)

  tag1 = 5678
  if rank == source:
      x[1] = 2.0
      ierr = comm.send(x[1], dest, tag1)
      print("processor ", rank, " sent x(1) = ", x[1])

  if rank == dest:
      ierr = comm.recv(x[1], source, tag1)
      print("processor ", rank, " got  x(1) = ", x[1])


  tagx = 4321
  if rank == source:
      ierr = comm.send(y, dest, tagx)
      print("processor ", rank, " sent ", y)

  if rank == dest:
      ierr = comm.recv(y, source, tagx)
      print("processor ", rank, " got  ", y)

  tag1 = 8765
  if rank == source:
      y[1,1] = 2.0
      ierr = comm.send(y[1,1], dest, tag1)
      print("processor ", rank, " sent y(1,1) = ", y[1,1])

  if rank == dest:
      ierr = comm.recv(y[1,1], source, tag1)
      print("processor ", rank, " got  y(1,1) = ", y[1,1])

  tag1 = 6587
  if rank == source:
      y[1,:] = 2.0
      ierr = comm.send(y[1,:], dest, tag1)
      print("processor ", rank, " sent y(1,:) = ", y[1,:])

  if rank == dest:
      ierr = comm.recv(y[1,:], source, tag1)
      print("processor ", rank, " got  y(1,:) = ", y[1,:])

  ierr = mpi_finalize()

compile the file and execute it using::

    pyccel --language="fortran" --compiler=mpif90 --filename=tests/examples/mpi/ex_1.py
    mpirun -n 2 tests/examples/mpi/ex_1

the result is::

   processor            0  sent    1.0000000000000000        1.0000000000000000        1.0000000000000000        1.0000000000000000     
   processor            0  sent x(1) =    2.0000000000000000     
   processor            0  sent    1.0000000000000000        1.0000000000000000        1.0000000000000000        1.0000000000000000        1.0000000000000000        1.0000000000000000     
   processor            0  sent y(1,1) =    2.0000000000000000     
   processor            1  got     1.0000000000000000        1.0000000000000000        1.0000000000000000        1.0000000000000000     
   processor            1  got  x(1) =    2.0000000000000000     
   processor            1  got     1.0000000000000000        1.0000000000000000        1.0000000000000000        1.0000000000000000        1.0000000000000000        1.0000000000000000     
   processor            1  got  y(1,1) =    2.0000000000000000     
   processor            0  sent y(1,:) =    2.0000000000000000        2.0000000000000000     
   processor            1  got  y(1,:) =    2.0000000000000000        2.0000000000000000

other examples can be found in **tests/examples/mpi**.

Tensor
******

When using the following import

.. code-block:: python

  from pyccel.mpi import *

**Pyccel** will convert every **Tensor** object to **MPI_Tensor** and thus allows for automatic parallelization of every loop.


Let's take the following example (**tests/examples/mpi/ex_16.py**)

.. code-block:: python

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

  mesh = tensor(r_x, r_y)

  u = zeros(mesh, double)

  #Initialization
  x = 0.0
  y = 0.0
  for i,j in mesh:
      x = i*hx
      y = j*hy

      u[i, j] = x*y*(x-1.0)*(y-1.0)

  #Communication
  sync(mesh) u

  #Delete memory
  del mesh

  ierr = mpi_finalize()

**mesh** is then constructed in parallel and relative indices are stored as **Range** objects. Since the loop is over the ranges of **mesh**, it will be automatically done on the corresponding distributed range.

Let's take a look at the fortran code:

.. code-block:: fortran

  program main
  use MPI
  implicit none
  integer :: mesh_ndim
  integer, allocatable :: mesh_dims (:)
  logical, allocatable :: mesh_periods (:)
  logical :: mesh_reorder
  integer :: mesh_comm_cart
  integer :: mesh_rank_in_cart
  integer, allocatable :: mesh_coords (:)
  integer, allocatable :: mesh_neighbor (:)
  integer :: mesh_sx
  integer :: mesh_ex
  integer :: mesh_sy
  integer :: mesh_ey
  integer :: mesh_line
  integer :: mesh_column
  integer, allocatable :: mesh_pads (:)
  integer :: mesh_tag_722050
  integer, dimension(MPI_STATUS_SIZE) :: i_mpi_status
  real(kind=8) :: hx
  real(kind=8) :: hy
  integer :: ntx
  integer :: i
  integer :: j
  integer :: ierr
  integer :: i_mpi_error
  real(kind=8), allocatable :: u (:, :)
  real(kind=8) :: y
  real(kind=8) :: x
  integer :: nty

  !  
  call mpi_init(ierr)
  ntx = 8
  nty = 8
  ! Grid spacing 
  hx = 1.0d0*1.0d0/(1 + ntx)
  hy = 1.0d0*1.0d0/(1 + nty)
  mesh_ndim = 2
  allocate(mesh_dims(0:mesh_ndim-1)); mesh_dims = 0
  mesh_dims(0) = 2
  mesh_dims(1) = 2
  allocate(mesh_periods(0:mesh_ndim-1)); mesh_periods = .False.
  mesh_periods(0) = .False.
  mesh_periods(1) = .False.
  mesh_reorder = .False.
  call mpi_cart_create (MPI_comm_world, mesh_ndim, mesh_dims, mesh_periods &
        , mesh_reorder, mesh_comm_cart, i_mpi_error)
  call mpi_comm_rank (mesh_comm_cart, mesh_rank_in_cart, i_mpi_error)
  allocate(mesh_coords(0:mesh_ndim-1)); mesh_coords = 0
  call mpi_cart_coords (mesh_comm_cart, mesh_rank_in_cart, mesh_ndim, &
        mesh_coords, i_mpi_error)
  allocate(mesh_neighbor(0:2*mesh_ndim-1)); mesh_neighbor = 0
  call mpi_cart_shift (mesh_comm_cart, 0, 1, mesh_neighbor(3), &
        mesh_neighbor(1), i_mpi_error)
  call mpi_cart_shift (mesh_comm_cart, 1, 1, mesh_neighbor(2), &
        mesh_neighbor(0), i_mpi_error)
  mesh_sx = mesh_coords(0)*ntx/mesh_dims(0)
  mesh_ex = (mesh_coords(0) + 1)*ntx/mesh_dims(0)
  mesh_sy = mesh_coords(1)*nty/mesh_dims(1)
  mesh_ey = (mesh_coords(1) + 1)*nty/mesh_dims(1)
  call MPI_type_vector (1 + mesh_ey - mesh_sy, 1, 3 + mesh_ex - mesh_sx, &
        MPI_DOUBLE, mesh_line, i_mpi_error)
  call MPI_type_commit (mesh_line, i_mpi_error)
  call MPI_type_contiguous (1 + mesh_ex - mesh_sx, MPI_DOUBLE, mesh_column &
        , i_mpi_error)
  call MPI_type_commit (mesh_column, i_mpi_error)
  allocate(mesh_pads(0:mesh_ndim-1)); mesh_pads = 0
  mesh_pads(0) = 1
  mesh_pads(1) = 1
  mesh_tag_722050 = 722050
  allocate(u(-mesh_pads(0) + mesh_sx:mesh_pads(0) + mesh_ex, -mesh_pads(1 &
        ) + mesh_sy:mesh_pads(1) + mesh_ey)); u = 0.0
  ! Initialization 
  x = 0.0d0
  y = 0.0d0
  do i = mesh_sx, -1 + mesh_ex, 1
    do j = mesh_sy, -1 + mesh_ey, 1

      x = i*hx
      y = j*hy
      u(i, j) = x*y*(-1.0d0 + x)*(-1.0d0 + y)
      ! Communication
    end do
  end do


  call mpi_sendrecv (u(mesh_sx, mesh_sy), 1, mesh_line, mesh_neighbor(0), &
        mesh_tag_722050, u(1 + mesh_ex, mesh_sy), 1, mesh_line, &
        mesh_neighbor(2), mesh_tag_722050, mesh_comm_cart, i_mpi_status, &
        i_mpi_error)

  call mpi_sendrecv (u(mesh_ex, mesh_sy), 1, mesh_line, mesh_neighbor(2), &
        mesh_tag_722050, u(-1 + mesh_sx, mesh_sy), 1, mesh_line, &
        mesh_neighbor(0), mesh_tag_722050, mesh_comm_cart, i_mpi_status, &
        i_mpi_error)

  call mpi_sendrecv (u(mesh_sx, mesh_sy), 1, mesh_column, mesh_neighbor(3) &
        , mesh_tag_722050, u(mesh_sx, 1 + mesh_ey), 1, mesh_column, &
        mesh_neighbor(1), mesh_tag_722050, mesh_comm_cart, i_mpi_status, &
        i_mpi_error)

  call mpi_sendrecv (u(mesh_sx, mesh_ey), 1, mesh_column, mesh_neighbor(1) &
        , mesh_tag_722050, u(mesh_sx, -1 + mesh_sy), 1, mesh_column, &
        mesh_neighbor(3), mesh_tag_722050, mesh_comm_cart, i_mpi_status, &
        i_mpi_error)
  ! Delete memory 
  deallocate(mesh_pads)
  deallocate(mesh_neighbor)
  deallocate(mesh_coords)
  deallocate(mesh_periods)
  deallocate(mesh_dims)
  call mpi_comm_free (mesh_comm_cart, i_mpi_error)
  call mpi_finalize(ierr)

  end


