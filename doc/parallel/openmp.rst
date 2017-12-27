OpenMP
******

There are two levels to work with **OpenMP**, called *level-0* and *level-1*.

level-0
^^^^^^^

This allows to write valid *OpenMP* instructions and are handled in two steps:

* in the grammar, in order to parse the *omp* pragams

* as a Pyccel header. Therefor, you can import and call *OpenMP* functions as you would do it in *Fortran* or *C*.

Examples
________

.. literalinclude:: ../../tests/scripts/openmp/core/ex1.py 
  :language: python

See :download:`script <../../tests/scripts/openmp/core/ex1.py>`.

Now, run the command::

  pyccel tests/scripts/openmp/core/ex1.py --openmp
  export OMP_NUM_THREADS=4

Executing the associated binary gives::

   > threads number :            1
   > maximum available threads :            4
   > thread  id :            0
   > thread  id :            3
   > thread  id :            1
   > thread  id :            2

See more `OpenMP examples`_.

.. _OpenMP examples: https://github.com/ratnania/pyccel/tree/master/tests/scripts/openmp/core

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


from pyccel.stdlib.parallel.openmp import Range
from pyccel.stdlib.parallel.openmp import Parallel

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

