Rules
*****

In this section, we provide a set of rules that were used to ensure the **one-to-one** correspondance between *Python* and *Fortran*. These rules are applied while annotating the *AST*.

.. note:: the first letter describes the message nature (W: warning, E: error, ...)

.. note:: the second letter describes the related section (F: function, E: expression, ...)

.. todo:: bounds check


.. Errors
.. ^^^^^^

- **EF001**: a function with with at least one argument or returned value, must have an associated header 



- **ES001**: **Except** statement is not covered by pyccel 
- **ES002**: **Finally** statement is not covered by pyccel 
- **ES003**: **Raise** statement is not covered by pyccel 
- **ES004**: **Try** statement is not covered by pyccel 
- **ES005**: **Yield** statement is not covered by pyccel 

.. Warnings
.. ^^^^^^^^

- **WF001**: all returned arguments must be atoms, no expression is allowed

- **WF002**: a returned variable should appear in the function header. Otherwise, *Type Inference* is used.


- **WC001**: class defined without **__del__** method. It will be added automatically  

Syntax
^^^^^^

- class attributes are defined as *DottedName* objects. This means that an expression

  .. code-block:: python

    self.x = x

  can be *printed*, while

  .. code-block:: python

    self.x = self.y

  will lead to an error, since *sympy* can not manipulate *DottedName*, which is suppose to be a *string* and not a *Symbol*.
  This is fixed in the *semantic* stage by converting *Symbol* objects to *Variable*.

Semantic
^^^^^^^^

Type inference
______________

Functions
_________

- a function that has no argument and no result does not need a header.

- Functions with no parameterized arguments must have a header 

- Functions with only parameterized arguments does not need a header 

- all returned arguments must be in general atoms, no expression is allowed 

- results are computed at the decoration stage
