pyccel
======

**Pyccel** stands for Python extension language using accelerators.

The aim of **Pyccel** is to provide a simple way to generate automatically, parallel low level code. The main uses would be:

1. Convert a *Python* code into a low-level target language (Fortran or C)

2. Accelerate a selected *Python* code, by using a pre-process, where the original code is translated into a low-level language, then compiled using **f2py** for example.

In order to achieve these tasks, in **Pyccel** we deal with the following points:

a. Implement a new *Python* parser (we do not need to cover all *Python* grammar)

b. Enrich *Python* with new statments to provide multi-threading (although some of them already exist) at the target level

c. Extends the concepts presented in **sympy** allowing for automatic code generation.  

Install
*******

run::

  python setup.py install --prefix=MY_INSTALL_PATH

this will install a *python* library **pyccel** and a *binary* called **pyccel**.

If **prefix** is not given, you will need to be in *sudo* mode. Otherwise, you will need to update your *.bashrc* or *.bash_profile* file with::

  export PYTHONPATH=MY_INSTALL_PATH/lib/python2.7/site-packages/:$PYTHONPATH
  export PATH=MY_INSTALL_PATH/bin:$PATH

for tests, run::

  python tests/run_tests.py
  python tests/run_tests_openmp.py

Documentation
*************

Can be found `here <http://ratnani.org/documentations/pyccel/>`_

Examples
********

Hello World
^^^^^^^^^^^

Let us consider the following *Python* file (*helloworld.py*)

.. code-block:: python

  def helloworld():
      print("* Hello World!!")

Now, run the command::

  pyccel --language="fortran" --compiler="gfortran" --filename=helloworld.py

The generated *Fortran* code is

.. code-block:: fortran

  module pyccel_m_helloworld

  implicit none

  contains
  ! ........................................
  subroutine helloworld()
  implicit none

  print *, '* Hello World!!'

  end subroutine
  ! ........................................


  end module pyccel_m_helloworld

Functions and Subroutines
^^^^^^^^^^^^^^^^^^^^^^^^^

Let's take a look at the file *tests/examples/ex5.py*, listed below

.. code-block:: python

  def f(u,v):
      t = u - v
      return t

  def g(x,v):
      m = x - v
      t =  2.0 * m
      z =  2.0 * t
      return t, z

  x = 1.0
  y = 2.0

  w    = 2 * f(x,y) + 1.0
  z, t = g(x,w)

  print(z)
  print(t)

Now, run the command::

  pyccel --language="fortran" --compiler="gfortran" --filename=tests/examples/ex5.py --execute

This will parse the *Python* file, generate the corresponding *Fortran* file, compile it and execute it. The result is::

   4.00000000    
   8.00000000 

Now, let us take a look at the *Fortran* file

.. code-block:: fortran

  program main

  implicit none
  real :: y
  real :: x
  real :: z
  real :: t
  real :: w

  !  
  x = 1.0d0
  y = 2.0d0
  w = 2*f(x, y) + 1.0d0
  call g (x, w, z, t)
  print * ,z
  print * ,t

  contains
  ! ........................................
  real function f(u, v)  result(t)
  implicit none
  real, intent(in)  :: u
  real, intent(in)  :: v

  t = u - v

  end function
  ! ........................................

  ! ........................................
  subroutine g(x, v, t, z)
  implicit none
  real, intent(out)  :: t
  real, intent(out)  :: z
  real, intent(in)  :: x
  real, intent(in)  :: v
  real :: m

  m = -v + x
  t = 2.0d0*m
  z = 2.0d0*t

  end subroutine
  ! ........................................


  end

Matrix-Matrix product
^^^^^^^^^^^^^^^^^^^^^

Let's take a look at the file *tests/matrix_product.py*, listed below

.. code-block:: python

  n = 2
  m = 4
  p = 2

  a = zeros(shape=(n,m), dtype=float)
  b = zeros(shape=(m,p), dtype=float)
  c = zeros(shape=(n,p), dtype=float)

  for i in range(0, n):
      for j in range(0, m):
          a[i,j] = i-j

  for i in range(0, m):
      for j in range(0, p):
          b[i,j] = i+j

  for i in range(0, n):
      for j in range(0, p):
          for k in range(0, p):
              c[i,j] = c[i,j] + a[i,k]*b[k,j]

  print(c)

Now, run the command::

  pyccel --language="fortran" --compiler="gfortran" --filename=tests/matrix_product.py --execute

This will parse the *Python* file, generate the corresponding *Fortran* file, compile it and execute it. The result is::

  -1.00000000       0.00000000      -2.00000000       1.00000000

Now, let us take a look at the *Fortran* file

.. code-block:: fortran

  program main

  implicit none
  real, allocatable :: a (:, :)
  real, allocatable :: c (:, :)
  real, allocatable :: b (:, :)
  integer :: i
  integer :: k
  integer :: j
  integer :: m
  integer :: n
  integer :: p

  !  
  ! from numpy import zeros 
  n = 2
  m = 4
  p = 2
  allocate(a(0:n-1, 0:m-1)) ; a = 0
  allocate(b(0:m-1, 0:p-1)) ; b = 0
  allocate(c(0:n-1, 0:p-1)) ; c = 0
  do i = 0, n - 1, 1
      do j = 0, m - 1, 1
          a(i, j) = i - j
      end do
  end do
  do i = 0, m - 1, 1
      do j = 0, p - 1, 1
          b(i, j) = i + j
      end do
  end do
  do i = 0, n - 1, 1
      do j = 0, p - 1, 1
          do k = 0, p - 1, 1
              c(i, j) = a(i, k)*b(k, j) + c(i, j)
          end do
      end do
  end do
  print * ,c

  end

Openmp examples
^^^^^^^^^^^^^^^

Matrix-Matrix product
_____________________

Let's take a look at the file *tests/examples/openmp/matrix_product.py*, listed below

.. code-block:: python

  from numpy import zeros

  n = 2000
  m = 4000
  p = 2000

  a = zeros(shape=(n,m), dtype=float)
  b = zeros(shape=(m,p), dtype=float)
  c = zeros(shape=(n,p), dtype=float)

  x = 0
  y = 0

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

  pyccel --language="fortran" --compiler="gfortran" --openmp --filename=tests/examples/openmp/matrix_product.py

This will parse the *Python* file, generate the corresponding *Fortran* file and compile it. 

.. note:: **Openmp** is activated using the flag **--openmp** in the command line.

The generated *Fortran* code is

.. code-block:: fortran

  program main
  use omp_lib 
  implicit none
  real, allocatable :: a (:, :)
  real, allocatable :: c (:, :)
  real, allocatable :: b (:, :)
  integer :: i
  integer :: k
  integer :: j
  integer :: m
  integer :: n
  integer :: p

  !  
  ! from numpy import zeros 
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

The following plot shows the scalability of the generated code on **LRZ** using :math:`(n,m,p) = (5000,7000,5000)`.

.. figure:: doc/include/openmp/matrix_product_scalability.png 
   :align: center
   :scale: 25% 

   Weak scalability on LRZ. CPU time is given in seconds.

.. figure:: doc/include/openmp/matrix_product_speedup.png 
   :align: center
   :scale: 25% 

   Speedup on LRZ

TODO
****

- improve precision

- **elif** statements

- pointers

- structures and classes

- procedure interfaces

- user *Fortran/c* functions provided as inputs

- BLAS

- LAPACK

- symbolic expressions (find a way to use directly functions that are defined in *sympy*)
