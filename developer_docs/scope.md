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

3.  All newly created variables must be inserted into the scope using the `insert_variable` function. In order to be able to locate the variable later the second argument of the `insert_variable` function (the name of the variable in the Python code) should be provided.

Temporary variables can also be created using the `get_temporary_variable` function. In this case it is not necessary to use the `insert_variable` function.

## Locating objects in the scope

The `find` function allows objects to be collected from the scope after they have been inserted. The function uses the `PyccelSymbol` (originating in the syntactic stage and therefore matching the Python name of the object) to locate the associated variable/class/etc (originating in the semantic stage and therefore matching the new name of the object). This difference in naming is why it is important to provide the Python name to the `insert_variable` function. The `find` function can be accelerated a little by only looking in the category of object that we are expecting (e.g. variables not classes).

In the `SemanticParser` some helper functions are provided to wrap `Scope.find`. For example, variables can be located using the functions `get_variable` or `check_for_variable`. The former looks for variables and raises an error if they are not found, while the latter checks if a variable exists in the scope (or any enclosing scopes). If it exists then it is returned, if not `None` is returned.

Additionally, there are some functions in the scope which can be used to get all the existing objects. Here is a subset, more information can be found in the docstrings of the class:
-   Each category of object has an property which allows the user to access the objects of that category in the scope (e.g. `Scope.variables`) which can be used to get, for example, all the variables in the current scope (this function only relates to the exact scope not the enclosing scopes).
-   The function `Scope.local_used_symbols` provides access to all objects in the local scope (i.e. excluding enclosing scopes). This is mostly useful when initialising scopes after the syntactic stage. These scopes need to be aware of the symbols which were discovered in the syntactic stage and the new names that were chosen to avoid collisions.
-   The function `Scope.all_used_symbols` provides a set of all symbols which will be available in this scope in the generated code (this includes symbols inherited from enclosing scopes). This set is very useful when finding new names which will not collision with existing symbols.
