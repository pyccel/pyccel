.. highlight:: rst

.. _restrictions:

Python Restrictions
*******************

Typing restrictions
^^^^^^^^^^^^^^^^^^^

Native *Python* objects are **implicitly** typed. This means that the following instructions are valid since the assigned *rhs* has a *static* type.

.. code-block:: python

  x = 1                   # OK
  y = 1.                  # OK
  s = 'hello'             # OK
    
Variables should only have one type so the following statements are invalid.

.. code-block:: python

  a = 1                   # OK
  a = '1'                 # KO
  b = None                # KO
  
Concerning *lists* and *tuples*, all their elements must be of the same type.

.. code-block:: python

  z = [1, 4, 9]           # OK
  t = (1., 4., 9., 16.0)  # OK

  z = [1, 4, 'a']         # KO
  t = (1., 4., 9., [])    # KO
  


  
Subset Limitations
^^^^^^^^^^^^^^^^^^
* multiple inheritance
* nested classes
* argument unpacking (*args and **kwargs) 
* Try,Except and Finaly block
* yield keyword and generators
* sets and dictionaries



