.. highlight:: rst

.. _restrictions:

Python Restrictions
*******************

Native *Python* objects are **implicitly** typed. This means that the following instructions are valid since the assigned *rhs* has a *static* type.

.. code-block:: python

  x = 1                   # OK
  y = 1.                  # OK
  s = 'hello'             # OK
  z = [1, 4, 9]           # OK
  t = (1., 4., 9., 16.0)  # OK

Concerning *lists* and *tuples*, all their elements must be of the same type.

.. code-block:: python

  z = [1, 4, 'a']         # KO
  t = (1., 4., 9., [])    # KO
