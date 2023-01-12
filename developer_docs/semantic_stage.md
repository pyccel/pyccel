# Semantic Stage

The semantic stage is described by the file [pyccel.parser.semantic](../pyccel/parser/semantic.py).

The semantic stage serves several purposes:
1.  [**Types**](#Types) : Determine the type of each symbol
2.  [**Impose Restrictions**](#Impose-restrictions) : Ensure that the code follows any restrictions that Pyccel imposes
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
-   **data type** : boolean/integer/float/complex/class type/etc
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

-   _symbol_ : The Python AST object should be passed here. This object contains information about its position in the file (line number, column) which ensures the user can more easily locate their error

-   _severity_ : The severity level must be one of the following:
    -   _warning_ : An error will be printed but Pyccel will continue executing
    -   _error_ : An error will be printed but Pyccel will continue executing the syntactic stage
    -   _fatal_ : An error will be printed and Pyccel will stop executing. This is the level used the most frequently in the semantic stage as, if information such as the type cannot be determined, the object cannot be created which will cause problems later in the execution.

It should be noted that Pyccel assumes that the user has provided valid code.
It is not feasible to provide error messages for every possible coding error so we limit ourselves to simple validity checks and Pyccel restriction errors.

## Function Recognition

Function calls are split into two groups which are handled in different ways.

The simplest case is the case where the function is defined by the user.
In this case the `funcdef` attribute of the [`pyccel.ast.core.FunctionCall`](../pyccel/ast/core.py) class should be an object of type [`pyccel.ast.core.FunctionDef`](../pyccel/ast/core.py).
The type of the `FunctionCall` (required to assign the result to a `Variable`) is easily determined from the type of the result variable(s).

The second case involves functions that are recognised by Pyccel.
This includes Python [built-in functions](https://docs.python.org/3/library/functions.html) and functions imported from supported libraries (`numpy`, `math`, etc).
Built-in functions do not need importing.
Instead they are recognised via the function [`pyccel.ast.utilities.builtin_function`](../pyccel/ast/utilities.py) which uses the dictionary [`pyccel.ast.builtins.builtin_functions_dict`](../pyccel/ast/builtins.py) to identify supported functions.
Functions from supported libraries are saved in an object of type [`pyccel.ast.core.PyccelFunctionDef`](../pyccel/ast/core.py) when they are imported.
These functions are handled one of two ways.
If there is special treatment which requires functions from the `SemanticParser` then a `_visit_X` function should be created.
The `SemanticParser._visit_FunctionCall` function will call this visitation function internally if it exists.
Otherwise the object will be created in the `SemanticParser._handle_function` function and its type will be determined by its constructor.

## Imports

In order for the semantic parser to have all information necessary for the different steps, imports must be handled correctly.
In the case of user-defined modules the imported file must have also been parsed.
The class [pyccel.parser.parser.Parser](../pyccel/parser/parser.py) ensures that functions are parsed in the correct order so that the information is available.
In the `_visit_Import` function, the [`pyccel.ast.core.Module`](../pyccel/ast/core.py) object representing the imported file is therefore available.
This object contains all necessary `FunctionDef`, `Variable` and `ClassDef` objects which may be imported.
These are then placed into the `Scope.imports` dictionary so they can be recognised when they are used in the file.

The case of modules supported by Pyccel is somewhat simpler.
In this case there should be an associated file `pyccel/ast/moduleext.py` (e.g. `numpyext.py`, `itertoolsext.py`) containing all the AST nodes related to this module.
The file should also contain a [`pyccel.ast.core.Module`](../pyccel/ast/core.py), listing all the objects which are in the file.
The `Module` object must then be saved in the [`pyccel.ast.utilities.builtin_import_registery`](../pyccel/ast/utilities.py) dictionary.

## Low-level Objects

Python is a high-level language.
As a result many implementation details are hidden to the user.
In low-level languages it is necessary to know this information.
In the semantic stage we must therefore identify these details and add objects representing them.

The most obvious examples of this are array allocation and garbage collection.
However, there are other examples so it is important to consider whether there is anything happening under the hood when adding a new `_visit_X` function.
Often to handle these hidden details new variables must be created.
In this case it is important to use the scope to avoid name collisions.
These variables that are created should be tagged as `is_temp = True`.
This allows Pyccel to differentiate between variables which appear in the code and should be preserved at all costs, and variables which are created by Pyccel and may be omitted if it leads to cleaner code.

Additional objects can often appear in awkward places where they cannot be easily returned as a `CodeBlock`.
This is the case for example if the object is needed to properly define something inside the right hand side of an `Assign`.
As the right hand side of an Assign cannot be a `CodeBlock` the additional expressions must be collected outside the usual flow.
The attribute `SemanticParser._additional_exprs` exists to hold these expressions.
This object is a list of lists which is initialised and inserted in `_visit_CodeBlock`.
A list of lists is necessary in case a `CodeBlock` can be found inside another (e.g. `a = [i for i in range]` where a `CodeBlock` contains the `Assign`, but another exists inside the `For` loop).

In order to avoid problems arising from forgetfulness we try to add additional objects in the most general place possible.
For example, allocation occurs in the function `SemanticParser._assign_lhs_variable`.
Variable declarations are created in the printer when needed from the scope variables (this allows each language to place the decorators in the most appropriate location).

## Object Tree

All the objects in the Pyccel AST inherit from [`pyccel.ast.basic.Basic`](../pyccel/ast/basic.py).
This super-class stores information about how the various objects are related.
This allows the class to provide functions such as `get_user_nodes` (which returns all objects of a given type which use the node), `get_attribute_nodes` (which returns all objects of a given type which are used by the node), `is_attribute_of` (which indicates if the argument is used by the node), `is_user_of` (which indicates if the argument uses the node), and `substitute` (which allows all occurrences of an object in the node to be replaced by a different object).
See [`pyccel.ast.basic.Basic`](../pyccel/ast/basic.py) for more information about these functions and other useful utility functions.

The tree is constructed in `Basic.__init__` using the `_attribute_nodes` attribute to recognise the names of attributes which must be added to the tree.
Nevertheless the object tree should be considered in two situations.
Firstly, if the object is constructed and AST objects are then added to it (e.g. the member function `pyccel.ast.core.CodeBlock.insert2body` used for the garbage collector).
In this case the new object does not pass through the constructor of its user.
It is therefore important to call `set_current_user_node` on the new object to update the tree.
Secondly, if the object contains all necessary information after the syntactic stage (e.g. `pyccel.ast.core.Continue`) we may be tempted to return the object as is.
However if this were done there would be multiple user nodes from both the semantic and the syntactic stage.
For example, if we need to have access to the containing function we could do `expr.get_user_nodes(FunctionDef)`.
We expect that this only returns semantic objects if `expr` is a result of the semantic stage.
However if objects such as `pyccel.ast.core.Continue` are returned as is, then we would get access to both the syntactic and the semantic versions of the containing function without any way to distinguish between the two.
To avoid this it is important to call the `pyccel.ast.basic.Basic.clear_user_nodes` function to remove the syntactic objects from the tree.

## Name Collisions

Name collisions may occur when generating temporary variables.
The scope helps keep track of the variables and prevent name collisions.
See [Scope](scope.md) for more details.

If variables are created as described above in the [Types](#Types) section, they will be added to the scope.
It is also possible that they will be renamed to avoid collisions.
For this reason it is very important to use the [`pyccel.parser.scope.Scope.find`](../pyccel/parser/scope.py) function to access variables.
There are two helper functions in the `SemanticParser` to facilitate these  searches:
-   `SemanticParser.check_for_variable` which returns the variable if it exists and None if it doesn't
-   `SemanticParser.get_variable` which raises an error if the requested variable is not found
