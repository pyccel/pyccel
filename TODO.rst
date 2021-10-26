TODO
====

- add appropriates imports from stdlib in all scripts

- make sure that every script is python valide: run python .. --lint for all scripts 

- add tests/pyccel/ast/test_parallel.py to travis

- in Variable, rank should be a property that uses *shape*?

- improve dotprint for our ast objects

- remove fprint from ast classes (see numpyext)

- move all numpy objects to numpyext

- check the correctness of pure function

Bugs
****

- **lapack** run_tests is not working anymore

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

Parser
******

* improve assignable for value in AssignStmt

* improve function headers 

* improve AugAssign (improve it like what has been done for Assign)

* add precision, double complex

* improve debug mode and verbosity in parser and syntax

* check error messages (and exceptions in syntax) and improve their treatments

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

OOP
***

* inheritence


Parallel
********

MPI
^^^

* Cart 3d

* iterators

* communication, reduction, etc

OpenMP
^^^^^^

* in get_with_clauses and get_for_clauses, we must be careful about args and kwargs for the __init__ call

* check valid values for clause arguments

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
