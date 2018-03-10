Functions
*********

Unlike *c/c++*, *Python* functions do not have separate header files or interface/implementation sections like *Pascal/Fortran*.

A function is simply defined using::

  def f(u,v):
      t = u - v
      return t

As we are targeting a *strongly typed* language, unfortunately, the first thing to do is to add a *header* as the following::

  #$ header f(double, double) results(double)
  def f(u,v):
      t = u - v
      return t

this tells *Pyccel* that the input/output arguments are of *double precision* type.
