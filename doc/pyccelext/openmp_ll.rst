OpenMP
******

This allows to write valid *OpenMP* instructions and are handled in two steps:

* in the grammar, in order to parse the *omp* pragams

* as a Pyccel header. Therefor, you can import and call *OpenMP* functions as you would do it in *Fortran* or *C*.

Examples
^^^^^^^^

.. literalinclude:: ../../tests/scripts/openmp/core/ex1.py 
  :language: python

See :download:`script <../../tests/scripts/openmp/core/ex1.py>`.

Now, run the command::

  pyccel tests/scripts/openmp/core/ex1.py --openmp
  export OMP_NUM_THREADS=4

Executing the associated binary gives::

   > threads number :            1
   > maximum available threads :            4
   > thread  id :            0
   > thread  id :            3
   > thread  id :            1
   > thread  id :            2

See more `OpenMP examples`_.

.. _OpenMP examples: https://github.com/ratnania/pyccel/tree/master/tests/scripts/openmp/core

Matrix multiplication
_____________________

Let's take a look at the file *tests/examples/openmp/matrix_product.py*, listed below

.. code-block:: python

  from numpy import zeros

  n = 500
  m = 700
  p = 500

  a = zeros((n,m), double)
  b = zeros((m,p), double)
  c = zeros((n,p), double)

  #$ omp parallel
  #$ omp do schedule(runtime)
  for i in range(0, n):
      for j in range(0, m):
          a[i,j] = i-j
  #$ omp end do nowait

  #$ omp do schedule(runtime)
  for i in range(0, m):
      for j in range(0, p):
          b[i,j] = i+j
  #$ omp end do nowait

  #$ omp do schedule(runtime)
  for i in range(0, n):
      for j in range(0, p):
          for k in range(0, p):
              c[i,j] = c[i,j] + a[i,k]*b[k,j]
  #$ omp end do
  #$ omp end parallel

Now, run the command::

  pyccel tests/examples/openmp/matrix_product.py --compiler="gfortran" --openmp

This will parse the *Python* file, generate the corresponding *Fortran* file and compile it. 

.. note:: **Openmp** is activated using the flag **--openmp** in the command line.

The generated *Fortran* code is

.. code-block:: fortran

  program main
  use omp_lib 
  implicit none
  real(kind=8), allocatable :: a (:, :)
  real(kind=8), allocatable :: c (:, :)
  real(kind=8), allocatable :: b (:, :)
  integer :: i
  integer :: k
  integer :: j
  integer :: m
  integer :: n
  integer :: p

  !  
  n = 500
  m = 700
  p = 500
  allocate(a(0:n-1, 0:m-1)) ; a = 0
  allocate(b(0:m-1, 0:p-1)) ; b = 0
  allocate(c(0:n-1, 0:p-1)) ; c = 0
  !$omp parallel
  !$omp do schedule(runtime)
  do i = 0, n - 1, 1
    do j = 0, m - 1, 1
      a(i, j) = i - j
    end do
  end do
  !$omp end do  nowait
  !$omp do schedule(runtime)
  do i = 0, m - 1, 1
    do j = 0, p - 1, 1
      b(i, j) = i + j
    end do
  end do
  !$omp end do  nowait
  !$omp do schedule(runtime)
  do i = 0, n - 1, 1
    do j = 0, p - 1, 1
      do k = 0, p - 1, 1
        c(i, j) = a(i, k)*b(k, j) + c(i, j)
      end do
    end do
  end do
  !$omp end do
  !$omp end parallel

  end
