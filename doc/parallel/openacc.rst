OpenACC
*******

Follozing the same idea for **OpenMP**, there are two levels to work with **OpenACC**, called *level-0* and *level-1*.

level-0
^^^^^^^

This allows to write valid *OpenACC* instructions and are handled in two steps:

* in the grammar, in order to parse the *acc* pragams

* as a Pyccel header. Therefor, you can import and call *OpenACC* functions as you would do it in *Fortran* or *C*.

Examples
________

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

level-1
^^^^^^^

This is a high level that enables the use of *OpenACC* through simple instructions.

.. automodule:: pyccel.stdlib.parallel.openacc
.. currentmodule:: pyccel.stdlib.parallel.openacc

.. autoclass:: Range
   :members:
   :private-members:
   :special-members:

.. autoclass:: Parallel
   :members:
   :private-members:
   :special-members:


Example: Hello world
____________________

.. literalinclude:: ../../tests/scripts/openacc/helloworld.py 
  :language: python

See :download:`script <../../tests/scripts/openacc/helloworld.py>`.


Example: reduction
__________________

.. literalinclude:: ../../tests/scripts/openacc/reduce.py 
  :language: python

See :download:`script <../../tests/scripts/openacc/reduce.py>`.


