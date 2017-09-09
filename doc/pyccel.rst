Library
=======

Documentation
*************

Parser
^^^^^^

.. automodule:: pyccel.parser
   :members:

Syntax
^^^^^^

.. inheritance-diagram:: pyccel.syntax

.. automodule:: pyccel.syntax
   :members:

Codegen
^^^^^^^

.. automodule:: pyccel.codegen
   :members:

Abstract Syntax Tree (AST)
^^^^^^^^^^^^^^^^^^^^^^^^^^

Types used to represent a full function/module as an Abstract Syntax Tree.

Most types are small, and are merely used as tokens in the AST. A tree diagram
has been included below to illustrate the relationships between the AST types.

.. inheritance-diagram:: pyccel.types.ast

.. automodule:: pyccel.types.ast
   :members:

Printers
^^^^^^^^

.. automodule:: pyccel.printers.codeprinter
   :members:

.. automodule:: pyccel.printers.fcode
   :members:

.. automodule:: pyccel.printers.ccode
   :members:

.. automodule:: pyccel.printers.luacode
   :members:

Patterns
^^^^^^^^

.. automodule:: pyccel.patterns.utilities
   :members:

Calculus
^^^^^^^^

.. automodule:: pyccel.calculus.finite_differences
   :members:

OpenMP
^^^^^^

.. inheritance-diagram:: pyccel.openmp.syntax

.. automodule:: pyccel.openmp.syntax
   :members:

Complexity
^^^^^^^^^^

Arithmetic
__________

.. automodule:: pyccel.complexity.operation
   :members:

Memory
______

.. automodule:: pyccel.complexity.memory
   :members:

