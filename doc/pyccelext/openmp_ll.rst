OpenMP
******

This allows to write valid *OpenMP* instructions and are handled in two steps:

* in the grammar, in order to parse the *omp* pragams

* as a Pyccel header. Therefor, you can import and call *OpenMP* functions as you would do it in *Fortran* or *C*.

Examples
^^^^^^^^

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
