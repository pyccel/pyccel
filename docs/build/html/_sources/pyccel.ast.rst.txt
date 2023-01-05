pyccel.ast package
==================

Submodules
----------

pyccel.ast.basic module
-----------------------

.. automodule:: pyccel.ast.basic
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autofunction:: pyccel.ast.basic.iterable
.. autoclass:: pyccel.ast.basic.Immutable
.. autoclass:: pyccel.ast.basic.ScopedNode

pyccel.ast.bind\_c module
-------------------------

.. automodule:: pyccel.ast.bind_c
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

pyccel.ast.bitwise\_operators module
------------------------------------

.. automodule:: pyccel.ast.bitwise_operators
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.bitwise_operators.PyccelBitComparisonOperator
.. autoclass:: pyccel.ast.bitwise_operators.PyccelBitOperator

pyccel.ast.builtin\_imports module
----------------------------------

.. automodule:: pyccel.ast.builtin_imports
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

pyccel.ast.builtins module
--------------------------

.. automodule:: pyccel.ast.builtins
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.builtins.Lambda
.. autoclass:: pyccel.ast.builtins.PythonAbs
.. autoclass:: pyccel.ast.builtins.PythonComplexProperty
.. autoclass:: pyccel.ast.builtins.PythonSum

pyccel.ast.c\_concepts module
-----------------------------

.. automodule:: pyccel.ast.c_concepts
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

pyccel.ast.class\_defs module
-----------------------------

.. automodule:: pyccel.ast.class_defs
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

pyccel.ast.core module
----------------------

.. automodule:: pyccel.ast.core
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autofunction:: pyccel.ast.core.apply
.. autofunction:: pyccel.ast.core.Concatenate
.. autofunction:: pyccel.ast.core.Decorator
.. autofunction:: pyccel.ast.core.DottedFunctionCall
.. autofunction:: pyccel.ast.core.ErrorExit
.. autofunction:: pyccel.ast.core.Exit
.. autofunction:: pyccel.ast.core.FuncAddressDeclare
.. autofunction:: pyccel.ast.core.FunctionAddress
.. autofunction:: pyccel.ast.core.IfSection
.. autofunction:: pyccel.ast.core.Iterable
.. autofunction:: pyccel.ast.core.PyccelFunctionDef
.. autofunction:: pyccel.ast.core.Raise

pyccel.ast.cwrapper module
--------------------------

.. automodule:: pyccel.ast.cwrapper
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autofunction:: pyccel.ast.cwrapper.C_to_Python
.. autofunction:: pyccel.ast.cwrapper.Python_to_C
.. autoclass:: pyccel.ast.cwrapper.PyModule_AddObject
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

pyccel.ast.datatypes module
---------------------------

.. automodule:: pyccel.ast.datatypes
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.datatypes.NativeNil
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:


pyccel.ast.functionalexpr module
--------------------------------

.. automodule:: pyccel.ast.functionalexpr
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

pyccel.ast.headers module
-------------------------

.. automodule:: pyccel.ast.headers
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.headers.Template
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

pyccel.ast.internals module
---------------------------

.. automodule:: pyccel.ast.internals
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autofunction:: pyccel.ast.internals.symbols

pyccel.ast.itertoolsext module
------------------------------

.. automodule:: pyccel.ast.itertoolsext
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

pyccel.ast.literals module
--------------------------

.. automodule:: pyccel.ast.literals
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autofunction:: pyccel.ast.literals.convert_to_literal
.. autoclass:: pyccel.ast.literals.Literal
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

pyccel.ast.macros module
------------------------

.. automodule:: pyccel.ast.macros
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

pyccel.ast.mathext module
-------------------------

.. automodule:: pyccel.ast.mathext
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

pyccel.ast.numpy\_wrapper module
--------------------------------

.. automodule:: pyccel.ast.numpy_wrapper
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autofunction:: pyccel.ast.numpy_wrapper.array_type_check
.. autofunction:: pyccel.ast.numpy_wrapper.find_in_numpy_dtype_registry
.. autofunction:: pyccel.ast.numpy_wrapper.scalar_type_check

pyccel.ast.numpyext module
--------------------------

.. automodule:: pyccel.ast.numpyext
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.numpyext.NumpyArray
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.numpyext.NumpyAutoFill
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.numpyext.NumpyConjugate
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.numpyext.NumpyFabs
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.numpyext.NumpyHypot
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.numpyext.NumpyNewArray
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.numpyext.NumpyTranspose
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.numpyext.NumpyUfuncBase
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.numpyext.NumpyUfuncBinary
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.numpyext.NumpyUfuncUnary
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autofunction:: pyccel.ast.numpyext.process_dtype

pyccel.ast.omp module
---------------------

.. automodule:: pyccel.ast.omp
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

pyccel.ast.operators module
---------------------------

.. automodule:: pyccel.ast.operators
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.operators.PyccelArithmeticOperator
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.operators.PyccelBinaryOperator
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.operators.PyccelBooleanOperator
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.operators.PyccelComparisonOperator
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.operators.PyccelUnaryOperator
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autofunction:: pyccel.ast.operators.broadcast

pyccel.ast.scipyext module
--------------------------

.. automodule:: pyccel.ast.scipyext
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

pyccel.ast.sympy\_helper module
-------------------------------

.. automodule:: pyccel.ast.sympy_helper
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

pyccel.ast.sysext module
------------------------

.. automodule:: pyccel.ast.sysext
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

pyccel.ast.utilities module
---------------------------

.. automodule:: pyccel.ast.utilities
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.utilities.LoopCollection
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autofunction:: collect_loops
.. autofunction:: collect_relevant_imports
.. autofunction:: compatible_operation
.. autofunction:: expand_inhomog_tuple_assignments
.. autofunction:: expand_to_loops
.. autofunction:: get_function_from_ast
.. autofunction:: insert_fors
.. autofunction:: insert_index
.. autofunction:: recognised_source

pyccel.ast.variable module
--------------------------

.. automodule:: pyccel.ast.variable
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.variable.Constant
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.variable.HomogeneousTupleVariable
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
.. autoclass:: pyccel.ast.variable.InhomogeneousTupleVariable
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

Module contents
---------------

.. automodule:: pyccel.ast
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
