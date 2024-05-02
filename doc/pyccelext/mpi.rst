mpi
***


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
