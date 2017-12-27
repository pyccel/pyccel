Introduction
************

Types
^^^^^

Dynamic *vs* Static typing
__________________________

Since our aim is to generate code in a low-level language, which is in most cases of static typed, we will have to devise an alternative way to construct/find the appropriate type of a given variable. 
This can be done by including the concept of *constructors* or use specific *headers* to assist *Pyccel* in finding/infering the appropriate type.

Let's explain this more precisely; we consider the following code

.. code-block:: python

  n = 5
  x = 2.0 * n

In this example, **n** will be interprated as an **integer** while **x** will be a **double** number, so everything is fine.

The problem arises when using a function, like in the following example

.. code-block:: python

  def f(n):
    x = 2.0 * n
    return x

  n = 5
  x = f(n)

Now the question is what would be the signature of **f** if there was no call to it in the previous script?

To overcome this ambiguity, we rewrite our function as

.. code-block:: python

  #$ header f(int)
  def f(n):
    x = 2.0 * n
    return x

Such an implementation still makes sens inside *Python*. As you can see, the type of *x* is infered by analysing our *expressions*.

Built-in Types
______________

The following are the built-in types in **Pyccel**::

  int, float, double, complex, array

.. todo:: boolean and string expressions not tested yet

Built-in Functions
^^^^^^^^^^^^^^^^^^

Mathematical functions
______________________

Mathematical functions are ::

   'transpose'
   'len'
   'log'
   'exp'
   'cos'
   'sin'
   'sqrt'
   'abs'
   'sign'
   'csc'
   'sec'
   'tan'
   'cot'
   'asin'
   'acsc'
   'acos'
   'asec'
   'atan'
   'acot'
   'atan2'
   'factorial'
   'ceil'
   'pow'
   'dot'
   'min'
   'max'

.. todo:: add transpose

Built-in Constants
^^^^^^^^^^^^^^^^^^

Mathematical constants
______________________

The following constants are available::

   'pi'

Data Types
^^^^^^^^^^

.. todo:: strctures and classe are not yet available

File and Directory Access
^^^^^^^^^^^^^^^^^^^^^^^^^

.. todo:: file and directory access is not yet available 

Importing modules
^^^^^^^^^^^^^^^^^

Importing modules is not allowed. However, you can import objects that are defined inside a given module. 

Iterators
^^^^^^^^^

There are 3 kind of iterators:

1. One that performs on groups (MPI)
   - for the moment, only **MPI_Tensor** is available

2. One that performs on teams (OpenMP, OpenACC)
   - this can be done using **prange** inside a **parallel** block

3. One that performs on atoms (sequential)


Functions and Subroutines
^^^^^^^^^^^^^^^^^^^^^^^^^

Let's take a look at the file *tests/examples/ex5.py*, listed below

.. code-block:: python

  #$ header f(double, double)
  def f(u,v):
      t = u - v
      return t

  #$ header g(double, double)
  def g(x,v):
      m = x - v
      t =  2.0 * m
      z =  2.0 * t
      return t, z

  x1 = 1.0
  y1 = 2.0

  w    = 2 * f(x1,y1) + 1.0
  z, t = g(x1,w)

  print(z)
  print(t)

Openmp examples
^^^^^^^^^^^^^^^

Matrix multiplication using OpenMP
__________________________________

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

The following plot shows the scalability of the generated code on **LRZ** using :math:`(n,m,p) = (5000,7000,5000)`.

.. figure:: ../include/openmp/matrix_product_scalability.png 
   :align: center
   :scale: 25% 

   Weak scalability on LRZ. CPU time is given in seconds.

.. figure:: ../include/openmp/matrix_product_speedup.png 
   :align: center
   :scale: 25% 

   Speedup on LRZ



