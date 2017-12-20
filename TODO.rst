TODO
====

Exceptions and Errors
*********************

* implement StopIteration as a class in stdlib

Imports
*******

* metavars: modules, module_name

* openmp.pyh only if openmp is used

Parser
******

* improve assignable for value in AssignStmt

* improve function headers (raise Exception NotImplemented when we have to type inference)

* improve precision, add double complex

* **inout** arguments are not handled yet

* OOP

* improve debug mode and verbosity in parser and syntax

* check error messages (and exceptions in syntax) and improve their treatments

* symbolic expressions (find a way to use directly *sympy* expressions)

* *eval* statement

* Expression of strings

* Expression of booleans

AST
***

* upate *clone* method of Variable

* cleaning: PointerVariable, AllocatableVariable, Variable, IndexedVariable, IndexedElement

* private variables if name is of the form *_name*, in fcode we should add a prefix *p_name*. The used prefix should be given as an argument of doprint in codegen

Codegen
*******

* in fcode: use  self._get_statement for every statement

* improve debug mode and verbosity in fcode and codegen

* code inlining

* private variables if name is of the form *_name*, in fcode we should add a prefix *p_name*. The used prefix should be given as an argument of doprint in codegen

* in *load_extension*: need to improve the use of dep_libs (ex BLAS_LIBRARIES, LAPACK_LIBRARIES)

* improve print for strings (remove *"*)

Parallel
********

MPI
^^^

* have MPI as headers

* write a new *Pyccel* MPI class (instead of the current *sympy* implementation)

* improve import

OpenMP
^^^^^^

* improve parallel constructor arguments

* improve *prange* (add new arguments, nowait, etc)

* improve import


linear(list[ : linear-step])


Commands line
*************

* pyccel output_dir (for temporary files) should be *.pyccel* for pyccel extensions

* procedure interfaces

External
********

* BLAS

* LAPACK
