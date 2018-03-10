Syntax analysis
***************

We use RedBaron_ to parse the *Python* code. **BNF** grammars are used to parse *headers*, *OpenMP* and *OpenAcc*. This is based on the textX_ project.

.. _RedBaron: https://github.com/PyCQA/redbaron

.. _textX: https://github.com/igordejanovic/textX


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

