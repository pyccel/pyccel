OpenACC
*******

This allows to write valid *OpenACC* instructions and are handled in two steps:

* in the grammar, in order to parse the *acc* pragams

* as a Pyccel header. Therefor, you can import and call *OpenACC* functions as you would do it in *Fortran* or *C*.

Examples
^^^^^^^^

.. literalinclude:: ../../tests/scripts/openacc/core/ex1.py 
  :language: python

See :download:`script <../../tests/scripts/openacc/core/ex1.py>`.

Now, run the command::

  pyccel tests/scripts/openacc/core/ex1.py --compiler=pgfortran --openacc

Executing the associated binary gives::

  number of available OpenACC devices :            1
  type of available OpenACC devices   :            2

See more `OpenACC examples`_.

.. _OpenACC examples: https://github.com/ratnania/pyccel/tree/master/tests/scripts/openacc/core

