.. highlight:: rst

.. _datatypes:

Datatypes and Precision
***********************

Native types
^^^^^^^^^^^^

============  =================
 Python        Fortran
============  =================
 int32         integer(kind=4)
 int64         integer(kind=8)
 float32       real(kind=4)
 float64       real(kind=8)
 complex64     complex(kind=4)
 complex128    complex(kind=8)
 tuple         static array
 list          dynamic array
============  =================

- The following stamtements are all understood by Pyccel and will give the desired datatype and the precision.

  .. code-block:: python


    from numpy import int32, int64, int
    x1 = 1
    x2 = int(1)
    x3 = int32(1)
    x4 = int64(1)
    
    from numpy import float32, float64, float
    y1 = 1.
    y2 = float(1.)
    y3 = float32(1.)
    y4 = float64(1.)
    
    from numpy import complex, complex64, complex128
    z1 = 1. + 1j   
    z1 = complex(1.)
    z2 = complex64(1.)
    z3 = complex128(1.)

  this will be converted to 

  .. code-block:: fortran

    program prog_ex

    implicit none
        
    integer(kind=4) :: x1  
    integer(kind=4) :: x2  
    integer(kind=4) :: x3  
    integer(kind=8) :: x4  
    real(kind=8) :: y1  
    real(kind=8) :: y2  
    real(kind=4) :: y3  
    real(kind=8) :: y4  
    complex(kind=8) :: z1  
    complex(kind=8) :: z2  
    complex(kind=4) :: z3  
    complex(kind=8) :: z4  

    x1 = 1
    x2 = Int(1, 4)
    x3 = Int(1, 4)
    x4 = Int(1, 8)

    y1 = 1.0d0
    y2 = Real(1.0d0, 8)
    y3 = Real(1.0d0, 4)
    y4 = Real(1.0d0, 8)

    z1 = cmplx(1.0d0,1)
    z2 = cmplx(1.0d0, 0.0d0, 8)
    z3 = cmplx(1.0d0, 0.0d0, 4)
    z4 = cmplx(1.0d0, 0.0d0, 8)

    end program prog_ex

Arrays
^^^^^^^

- in order to allocate memory we use numpy functions (empty, zeros, ones ...etc) and the following python lines 
 

  .. code-block:: python

    from numpy import array
    from numpy import empty

    x = array([1, 2, 3])
    y = empty((10, 10))

  will be converted to 

  .. code-block:: fortran

    program prog_arrays

    implicit none

    integer(kind=4), allocatable, target :: x (:) 
    real(kind=8), allocatable :: y (:,:) 

    allocate(x(0:2))
    x = (/ 1, 2, 3 /)

    allocate(y(0:9, 0:9))

    end program prog_arrays

Dynamic *vs* Static typing
__________________________

Since our aim is to generate code in a low-level language, which is in most cases of static typed, we will have to devise an alternative way to construct/find the appropriate type of a given variable. 
This can be done by including the concept of *constructors* or use specific *headers* to assist *Pyccel* in finding/infering the appropriate type.

Let's explain this more precisely, we consider the following code

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

