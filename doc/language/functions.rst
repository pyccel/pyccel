Functions
*********

Unlike *c/c++*, *Python* functions do not have separate header files or interface/implementation sections like *Pascal/Fortran*.

A function is simply defined using:

.. code-block:: python

  def f(u,v):
      t = u - v
      return t

As we are targeting a *strongly typed* language, unfortunately, the first thing to do is to add a *header* as the following:

.. code-block:: python

  #$ header f(double, double) results(double)
  def f(u,v):
      t = u - v
      return t

this tells *Pyccel* that the input/output arguments are of *double precision* type.

You can then call **f** even in a given expression:

.. code-block:: python

  x1 = 1.0
  y1 = 2.0

  w    = 2 * f(x1,y1) + 1.0

You can also define functions with multiple *lhs* and call them as in the following example:

.. code-block:: python

  #$ header g(double, double)
  def g(x,v):
      m = x - v
      t =  2.0 * m
      z =  2.0 * t
      return t, z

  x1 = 1.0
  y1 = 2.0

  z, t = g(x1,y1)
