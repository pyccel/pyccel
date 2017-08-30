Standard Library
================

Numpy functions
***************

The following **numpy** functions are recognized as *elements* of **Pyccel**:

* **zeros**. The following statements should be valid

.. code-block:: python

  # input python code
  a = zeros(shape=64, dtype=float)

The generated code should look like

.. code-block:: fortran

  ! generated fortran code
  real, allocatable :: a(:)
  ...
  a = zeros(64) 

* **linspace**

.. code-block:: python

  # input python code
  a = linspace(0.0, 1.0, 100, dtype=float)

The generated code should look like

.. code-block:: fortran

  ! generated fortran code
  real, allocatable :: a(:)
  ...
  a = linspace(0.0, 1.0, 100)

* **zeroslike**

.. code-block:: python

  # input python code
  a = zeroslike(x)

The generated code should look like

.. code-block:: fortran

  ! generated fortran code
  real, allocatable :: a(:)
  ...
  a = zeroslike(x)

* **ones**

.. code-block:: python

  # input python code
  a = ones(shape=64, dtype=float)

The generated code should look like

.. code-block:: fortran

  ! generated fortran code
  real, allocatable :: a(:)
  ...
  a = ones(64) 

* **random**

.. code-block:: python

  # input python code
  a = array([1.0, 0.25, 0.7, 0.9], dtype=float)

The generated code should look like

.. code-block:: fortran

  ! generated fortran code
  real :: a(4)
  ...
  a = (/ 1.0, 0.25, 0.7, 0.9 /)

* **meshgrid**

.. code-block:: python

  # input python code
  x = meshgrid(u,v)

The generated code should look like

.. code-block:: fortran

  ! generated fortran code
  real, allocatable :: x(:,:)
  ...
  x = meshgrid(u,v)

* **array**

.. code-block:: python

  # input python code
  a = array([1.0, 0.25, 0.7, 0.9], dtype=float)

The generated code should look like

.. code-block:: fortran

  ! generated fortran code
  real :: a(4)
  ...
  a = (/ 1.0, 0.25, 0.7, 0.9 /)


Python standard library
***********************

* **range**

* **len**
