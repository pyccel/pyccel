# Wrapper stage

Python is written in C. In order to create a module which can be imported from Python, we therefore have to print a wrapper using the [Python-C API](https://docs.python.org/3/c-api/index.html). This code must call the Pyccel generated translation of the code. If that code was not generated in C then further wrappers are necessary in order to make the code compatible with C.

For example in Fortran generated code there are often array arguments. These do not exist in C, therefore three or more arguments must be passed to the function instead:
1.  The data
2.  The size(s) of the array in each dimension
3.  The stride(s) between elements in each dimension

The wrapper stage takes the AST describing the generated code as an input and returns an AST which makes that code available to a target language (C or Python).

The entry point for the class `Wrapper` is the function `wrap`.

## `_wrap`

The `_wrap` function internally calls a function named `_wrap_X`, where `X` is the type of the object.
These functions must have the form:
```python
def _print_ClassName(self, stmt):
    ...
    return Y
```
Each of these `_wrap_X` functions should internally call the `_wrap` function on each of the elements relevant to the wrapper to obtain AST objects which describe the same information, but in a way that is accessible from the target language.

## Name Collisions

While creating the wrapper, it is often necessary to create multiple variables to fully describe something which was a single object in the original code. In order to facilitate reading the code these objects should have names similar to that of the original variable. As a result a **lot** of care must be taken to avoid name collisions.

In general the following rules should be respected:
-   Any variables which we wish to be able to retrieve from the scope using the `Scope.find` function must be added to the scope with `Scope.insert_symbol`. We cannot do this with 2 variables with the same name so the names must come from the AST where collisions have already been removed.
-   Any names saved via `Scope.insert_symbol` must be accessed using `Scope.get_expected_name` (as the name may have been changed to avoid other collisions).
-   Any new names must be created using `Scope.get_new_name`

## Fortran To C

The Fortran to C wrapper wraps Fortran code to make it callable from C. This module relies heavily on [`iso_c_binding`](https://stackoverflow.com/tags/fortran-iso-c-binding/info). It is used to ensure that the functions can be called correctly and that types are compatible. We do not use `iso_fortran_binding` as it was only introduced in Fortran 2018 and we do not expect compiler support for this in the near future.

### Scalar module variables

Scalar module variables are already accessible from C so no extra work is done here.

### Array module variables

Arrays are not compatible with C. Instead the wrapper prints a function which returns a pointer to the module variable as well as it's size information to make the information available from C.

### Functions

In the simplest case a wrapper around a function takes the same arguments, calls the function and then returns the result. The only difference between this function and the original function is the body (where the original function is called) and the `bind(c)` tag which ensures that the function name is accessible from C and is not mangled.

More complex cases include functions with array arguments or results, optional variables and functions as arguments.

Array arguments and results are handled similarly to array module variables, by adding additional variables for the shape and stride information.

Optional arguments are passed as C pointers. An if/else block then determines whether the pointer is assigned or not. This can be quite lengthy, however it is unavoidable for compilation with intel, or nvidia. It is also unavoidable for arrays as it is important not to index an array (to access the strides) which is not present.

Finally the most complex cases such as functions as arguments are simply not printed. Instead these cases raise warnings or errors to alert the user that support is missing.
