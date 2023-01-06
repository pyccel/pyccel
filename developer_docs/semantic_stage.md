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
- **data type** : bool/int/float/complex/class type/etc
- **precision** : The number of bytes required to store an object of this data type
- **rank** : The number of dimensions of the array (0 for a scalar)
- **shape** : The number of elements in each dimension of the array (`()` for a scalar)
- **order** : The order in which the data is stored in memory. See [order docs](order_docs.md) for more details.

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

## Function Recognition

## Imports

## Low-level Objects

## Object Tree

## Name Collisions

Name collisions may occur when generating 
