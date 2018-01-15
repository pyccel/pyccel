TODO
====

Bugs
****

- **zeros_like** is not copying attribut data

- **scatter.py** not working anymore

- improve **ValuedVariable**

- **Del** for a list of variables

Commands
********

* add *--fflags* to **pyccel** command.

Exceptions and Errors
*********************

* implement StopIteration as a class in stdlib

Imports
*******

* metavars: modules, module_name

* openmp.pyh only if openmp is used

* improve imports by using groupby to gather imported things by their module, and have to process the module only once

Parser
******

* improve assignable for value in AssignStmt

* improve function headers (raise Exception NotImplemented when we have to type inference)

* improve AugAssign (improve it like what has been done for Assign)

* improve precision, add double complex

* **inout** arguments are not handled yet

* OOP: inheritence

* improve debug mode and verbosity in parser and syntax

* check error messages (and exceptions in syntax) and improve their treatments

* symbolic expressions (find a way to use directly *sympy* expressions)

* *eval* statement

* Expression of strings

* Expression of booleans

* add is_compatible_header for functions

* stencil and vector must use Variable and be of any datatype

AST
***

* upate *clone* method of Variable

* cleaning: PointerVariable, AllocatableVariable, Variable, IndexedVariable, IndexedElement

* private variables if name is of the form *_name*, in fcode we should add a prefix *p_name*. The used prefix should be given as an argument of doprint in codegen

Codegen
*******

* Matrix is not printed correctly (must check what is done in Sympy)

* improve *namespace* property. For the moment, we only create the definition for FunctionHeader (if not MethodHeaded)

* in fcode: use  self._get_statement for every statement and be careful to comments (and omp pragmas)

* improve debug mode and verbosity in fcode and codegen

* code inlining

* private variables if name is of the form *_name*, in fcode we should add a prefix *p_name*. The used prefix should be given as an argument of doprint in codegen

* in *load_extension*: need to improve the use of dep_libs (ex BLAS_LIBRARIES, LAPACK_LIBRARIES)

* improve print for strings (remove *"*)

Parallel
********

MPI
^^^

* Cart 3d

* ietrators

* communication, reduction, etc

OpenMP
^^^^^^

* in get_with_clauses and get_for_clauses, we must be careful about args and kwargs for the __init__ call

* check valid values for clause arguments

* improve import

Commands line
*************

* pyccel output_dir (for temporary files) should be *.pyccel* for pyccel extensions

* procedure interfaces

External
********

* BLAS

* LAPACK

* FFTW

* HDF5

* SUPERLU

* PASTIX

Symbolic Computation
********************

* treate **1d** and **3d** cases for **weak_formulation**

* maybe we should add the *dim* as argument of **lambdify**

* vector case for glt and weak formulation

* algebra for differential operators (so that we can do *dx(u+v)*)
