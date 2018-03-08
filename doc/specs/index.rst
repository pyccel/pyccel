.. _specs:

Specifications
==============

Native types
************

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

Python Restrictions
*******************

Native *Python* objects are **implicitly** typed. This means that the following instructions are valid since the assigned *rhs* has a *static* type.

.. code-block:: python

  x = 1                   # OK
  y = 1.                  # OK
  s = 'hello'             # OK
  z = [1, 4, 9]           # OK
  t = (1., 4., 9., 16.0)  # OK

Concerning *lists* and *tuples*, all their elements must be of the same type.

.. code-block:: python

  z = [1, 4, 'a']         # KO
  t = (1., 4., 9., [])    # KO

Syntax parsing
**************

We use RedBaron_ to parse the *Python* code. **BNF** grammars are used to parse *headers*, *OpenMP* and *OpenAcc*. This is based on the textX_ project.

.. _RedBaron: https://github.com/PyCQA/redbaron

.. _textX: https://github.com/igordejanovic/textX

Contents
********

.. toctree::
   :maxdepth: 1 

   highlevel
   lowlevel
   rules
   openmp
   openacc
