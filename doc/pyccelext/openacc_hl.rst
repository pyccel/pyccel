.. highlight:: rst

.. _openacc:

OpenACC
*******

Following the same idea for **OpenMP**, there are two levels to work with **OpenACC**, called *level-0* and *level-1*.

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


