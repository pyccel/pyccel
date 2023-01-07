# Semantic Stage

The semantic stage is described by the file [pyccel.parser.semantic](../pyccel/parser/semantic.py).

The semantic stage serves several purposes:
1.  [**Types**](#Types) : Determine the type of each symbol
2.  [**Impose Restrictions**](#Impose-restrictions) : Ensure that the code follows any restictions that Pyccel imposes
3.  [**Function Recognition**](#Function-recognition) : Identify any functions that are recognised by Pyccel
4.  [**Imports**](#Imports) : Identify imports
5.  [**Low-level Objects**](#Low-level-objects) : Create any objects which are hidden in high-level code, but must appear explicitly in low-level code
6.  [**Object Tree**](#Object-tree) : Ensure the object tree is correctly constructed
7.  [**Name Collisions**](#Name-collisions) : Ensure any potential name collisions are avoided

## Navigation

The entry point for the class `SemanticParser` is the function `annotate`.
This function is called from the constructor to examine an AST (Abstract Syntax Tree) object created by the [syntactic stage](.syntactic_stage.md).

The key line of the function `annotate` is the call to `self._visit(self.ast, **settings)`.
All elements of the tree must be visited.
Similarly to in the [syntactic stage](.syntactic_stage.md), the `_visit` function internally calls a function named `_visit_X`, where `X` is the type of the object.
These functions must have the form:
```python
def _visit_ClassName(self, stmt):
    ...
```
Each of these `_visit_X` functions should internally call the `_visit` function on each of the elements of the object to obtain annotated objects which are combined to finally create an annotated syntax tree.

## Types

Variables and objects which can be saved in variables (e.g. literals and arrays), are  characterised by their type.
The type indicates all the information that allows the object to be declared in a low-level language.
The interface to access these characteristics is defined in the super class [`pyccel.ast.basic.PyccelAstNode`](../pyccel/ast/basic.py).
The characteristics are:
-   **data type** : bool/int/float/complex/class type/etc
-   **precision** : The number of bytes required to store an object of this data type
-   **rank** : The number of dimensions of the array (0 for a scalar)
-   **shape** : The number of elements in each dimension of the array (`()` for a scalar)
-   **order** : The order in which the data is stored in memory. See [order docs](order_docs.md) for more details.

The type of the different objects is determined in 2 different places.

`Variable` objects are created in the `SemanticParser._visit_Assign` function.
Their type is determined from the type of the right hand side, which should be a `PyccelAstNode`.
The function `SemanticParser._infer_type` infers the type from the right hand side object and returns a dictionary describing the different characteristics.
This dictionary is passed to the function `SemanticParser._assign_lhs_variable` which should always be used to create variables as it runs various checks including the validity of the type (e.g checking if the datatype has changed).
In addition to the above characteristics `Variable` objects also have a few additional characteristics such as the `memory_location` which are also determined in the `SemanticParser._infer_type` function.

All other objects determine their type in the class definition.
Thus each function which Pyccel recognises and handles can take the expected arguments and internally determine its characteristics.
Errors are also raised at this point if the arguments do not match the expected restrictions, although care must be taken to ensure that these tests and the type determinations do not take place at the syntactic stage where they would cause failures.

## Impose Restrictions

Not all valid Python code can be translated by Pyccel.
This is for one of three reasons:
1.  Support has not yet been added but is planned (e.g. classes)

2.  It would be impractical to support the code in a low-level language. E.g. functions which return objects of different types :
    ```python
    def f(a : bool, b : bool):
        if a and b:
            return 1.0, False
        elif b:
            return 10, 4j
        elif a:
            return True
        else:
            return
    ```

3.  A conscious choice has been made to not support code as there is no way to obtain a performant translation.
    Pyccel is designed to handle HPC applications.
    Therefore if the code cannot be performant, we prefer to warn the user so they can fix it rather than generating slow code (e.g. inhomogeneous lists).

When the translation fails due to Pyccel restrictions it is important to raise a clear error.
Users may be surprised and annoyed to find that their functioning Python code will not translate.
In order to facilitate their interactions with Pyccel, we therefore want to make sure that errors fully explain the problems with their code, and point towards any potential documentation which may show them how to fix the code.

Error handling uses the classes in the file [pyccel.errors.errors](../pyccel/errors/errors.py).
Errors in the semantic stage should raise a `PyccelSemanticError`.
Where possible this should be done by accessing the `Errors()` singleton and calling the `report` function.
This function takes several arguments (see docstring for more details).
The most important arguments are:
-   _message_ : Describe the issue that lead to the error

-   _symbol_ : The Python ast object should be passed here. This object contains information about its position in the file (line number, column) which ensures the user can more easily locate their error

-   _severity_ : The severity level must be one of the following:
    -   _warning_ : An error will be printed but Pyccel will continue executing
    -   _error_ : An error will be printed but Pyccel will continue executing the syntactic stage
    -   _fatal_ : An error will be printed and Pyccel will stop executing. This is the level used the most frequently in the semantic stage as, if information such as the type cannot be determined, the object cannot be created which will cause problems later in the exection.

It should be noted that Pyccel assumes that the user has provided valid code.
It is not feasable to provide error messages for every possible coding error so we limit ourselves to simple validity checks and Pyccel restriction errors.

## Function Recognition

Function calls are split into two groups which are handled in different ways.

The simplest case is the case where the function is defined by the user.
In this case the `funcdef` attribute of the [`pyccel.ast.core.FunctionCall`](../pyccel/ast/core.py) class should be an object of type [`pyccel.ast.core.FunctionDef`](../pyccel/ast/core.py).
The type of the `FunctionCall` (required to assign the result to a `Variable`) is easily determined from the type of the result variable(s).

The second case involves functions that are recognised by Pyccel.
This includes built-in functions and functions imported from supported libraries (numpy, math, etc).
Built-in functions do not need importing.
Instead they are recognised via the function [`pyccel.ast.utilities.builtin_function`](../pyccel/ast/utilities.py) which uses the dictionary [`pyccel.ast.builtins.builtin_functions_dict`](../pyccel/ast/builtins.py) to identify supported functions.
Functions from supported libraries are saved in an object of type [`pyccel.ast.core.PyccelFunctionDef`](../pyccel/ast/core.py) when they are imported.
These functions are handled one of two ways.
If there is special treatment which requires functions from the `SemantcParser` then a `_visit_X` function should be created.
The `SemanticParser._visit_FunctionCall` function will call this visitation function internally if it exists.
Otherwise the object will be created in the `SemanticParser._handle_function` function and its type will be determined by its constructor.

## Imports

## Low-level Objects

## Object Tree

## Name Collisions

Name collisions may occur when generating 
