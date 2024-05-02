# How to approach issues

This file summarises basic approaches which should allow you to attempt some of the issues marked `good-first-issue`

## Adding New Functions

To add a new function:

-   Determine the functions in C/Fortran that are equivalent to the function you want to support (ideally aim to support as many arguments as seems feasible, check the Python standard for the latest description of the function)
-   Add a class to represent the function. The class should go in the appropriate file in the [ast](../pyccel/ast) folder. This function will probably inherit from [PyccelInternalFunction](../pyccel/ast/internals.py)
-   Ensure the function is recognised in the semantic stage by adding it to the appropriate dictionary (see the function `builtin_function` and the dictionary `builtin_import_registery` in [ast/utilities.py](../pyccel/ast/utilities.py)
-   Add the print functions for the 3 languages
-   Add tests in the folder `tests/epyccel`

## Language Specific Bug Fixes

-   Use the issue description to reproduce the bug
-   Add a test targeting this specific problem (which fails in the master branch but will pass after your fix)
-   By comparing the original Python code and the generated code, try to locate the problematic line and therefore the print function in the corresponding code generation file
-   Try to fix the problem
