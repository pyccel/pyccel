## Scope

In computer science, the _scope_ is the area of a program where an item (e.g. variable, function, etc.) is recognised. For example a variable defined in a function will not be recognised outside of that function, therefore the function defines its scope.

In Pyccel, a `Scope` is an object defined in [parser/scope.py](../pyccel/parser/scope.py) which represents this concept. It includes all the functions, imports, variables, and classes which are available at a given point in the code. It also contains pointers to nested and parent scopes.

Each of these objects must be inserted into the scope using and insert function.

## `ScopedAstNode`

Each scope is associated with a class, e.g. `FunctionDef`, `For`, `Module`. These classes inherit from the `ScopedAstNode` class. The scope associated with the class instance, is saved within the class. This makes the scope available when the class instance is available. This is important so as to correctly set the `scope` variable in the `SemanticParser` and the different `CodePrinter`s.

## Name collisions

The `Scope` object keeps track of all names used in the described scope. This means that it can be used to prevent name collisions. In order to do so a few steps must be respected:

1.  In the syntactic stage all symbols in the code must be added to the correct scope. This is done via the function `insert_symbol`

2.  All names of variables created by Pyccel must be created using one of the following functions defined in the Scope class:
    -   `get_expected_name` : Collect the name which will be used to create the Variable referenced in the argument. In most cases this operation will be the identity operation, but it ensures that name collisions are handled and that the Symbol has been correctly inserted into the Scope
    -   `get_new_name` : Get a new name with no collisions. A name can be requested and will be used if available
    -   `get_new_incremented_symbol` : Get a new name with no collisions following a pattern. This function keeps track of the index appended to the incremented string so it is most useful when creating multiple names with the same prefix

3.  All newly created variables must be inserted into the scope using the `insert_variable` function.

Temporary variables can also be created using the `get_temporary_variable` function. In this case it is not necessary to use the `insert_variable` function.
