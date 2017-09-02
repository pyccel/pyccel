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

  cd tests
  python run_tests.py

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
  w = 2.0d0*f(x, y) + 1.0d0
  call g (x, w, z, t)
  print * , z
  print * , t

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

  n = int()
  m = int()
  p = int()
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
  n = 2.0d0
  m = 4.0d0
  p = 2.0d0
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
  print * , c

  end

TODO
****

- add *math* functions: sign, ceil, log, exp, cos, sin, sqrt, abs, pi (must be added as a declaration)

- improve precision

- pointers

- structures and classes

- procedure interfaces

- user *Fortran/c* functions provided as inputs

- BLAS

- LAPACK

- must rename *Piecewise* into *If* if we want to use it from sympy

- symbolic expressions (find a way to use directly functions that are defined in *sympy*)

BUGS
****

- The following code

  .. code-block:: python

    n = int()
    n = 5

  gives

  .. code-block:: fortran

    n = int()
    n = 5.0d0

- **sqrt** function does work on an indexed variable. Must check the other math functions and function calls and more generally expressions.
