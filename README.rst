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

BUGS
****

TODO
****

Numpy functions
^^^^^^^^^^^^^^^

The following **numpy** functions are recognized as *elements* of **Pyccel**:

* **zeros**. THe following statements should be valid

.. code-block:: python

  a = zeros(shape=64, dtype=float)

* **linspace**

* **zeroslike**

* **ones**

* **random**

* **meshgrid**

* **array**

.. code-block:: python

  a = array([1.0, 0.25, 0.7, 0.9])

The generated code should look like

.. code-block:: fortran

  real :: a(4)
  ...
  a = (/ 1.0, 0.25, 0.7, 0.9 /)


will generate a *Tuple* in the **AST**

Python standard library
^^^^^^^^^^^^^^^^^^^^^^^

* **range**

* **len**
