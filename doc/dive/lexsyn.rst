Lexical conventions and Syntax
******************************

Static verification
^^^^^^^^^^^^^^^^^^^

For more details, we highly recommand to take a look at `Pylint documentation`_.

.. _Pylint documentation: https://pylint.readthedocs.io

Pylint messages
_______________

To understand what *Pylint* is trying to tell you, please take a look at `Pylint codes`_.

.. _Pylint codes: http://pylint-messages.wikidot.com/all-codes

Retrieving messages from Pylint
_______________________________

**Pyccel** uses the `parse library`_ to retrieve error messages from *Pylint*.

.. _parse library: https://pypi.python.org/pypi/parse

Grammar
^^^^^^^

In this section, we give the **BNF** used grammar for parsing *Python*, *openmp* and *openacc* codes.

.. literalinclude:: ../../pyccel/parser/grammar/pyccel.tx 

See :download:`script <../../pyccel/parser/grammar/pyccel.tx>`.

Imports
_______

.. literalinclude:: ../../pyccel/parser/grammar/imports.tx 

See :download:`script <../../pyccel/parser/grammar/imports.tx>`.

OpenMP
______

.. literalinclude:: ../../pyccel/parser/grammar/openmp.tx 

See :download:`script <../../pyccel/parser/grammar/openmp.tx>`.

OpenACC
_______

.. literalinclude:: ../../pyccel/parser/grammar/openacc.tx 

See :download:`script <../../pyccel/parser/grammar/openacc.tx>`.

