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

