OpenMP
******

There are two levels to work with **OpenMP**, called *level-0* and *level-1*.


level-1
^^^^^^^

This is a high level that enables the use of *OpenMP* through simple instructions.

.. automodule:: pyccel.stdlib.parallel.openmp
.. currentmodule:: pyccel.stdlib.parallel.openmp

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

.. literalinclude:: ../../tests/scripts/openmp/helloworld.py 
  :language: python

See :download:`script <../../tests/scripts/openmp/helloworld.py>`.


Example: matrix multiplication
______________________________

.. literalinclude:: ../../tests/scripts/openmp/matrix_multiplication.py 
  :language: python

See :download:`script <../../tests/scripts/openmp/matrix_multiplication.py>`.

