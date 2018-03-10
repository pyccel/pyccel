.. highlight:: rst

.. _datatypes:

Datatypes
*********

Native types
^^^^^^^^^^^^

======== ===============
 Python   Fortran
======== ===============
int       int
float     real(4)
float64   real(8)
complex   complex(16)
tuple     static array
list      dynamic array
======== ===============

- Default type for floating numbers is **double precision**. Hence, the following stamtement 

  .. code-block:: python

    x = 1.

  will be converted to 

  .. code-block:: fortran

    real(8) :: x

    x = 1.0d0

Slicing
^^^^^^^

- When assigning a slice of **tuple**, we must allocate memory before (tuples are considered as static arrays). Therefor, the following python code

  .. code-block:: python

    a = (1, 4, 9, 16)
    c = a[1:]

  will be converted to 

  .. code-block:: fortran

    integer :: a (0:3)
    integer, allocatable :: c(:)

    a = (/ 1, 4, 9, 16 /)
    c = allocate(c(1,3))
    c = a(1 : )

.. todo:: memory allocation within the scope of definition


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

