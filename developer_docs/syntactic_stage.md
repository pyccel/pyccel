# Syntactic Stage

The syntactic stage is described by the file [pyccel.parser.syntactic](../pyccel/parser/syntactic.py)

The syntactic stage serves 4 main purposes:
1.  [**Navigation and AST Creation**](#Navigation-and-ast-creation) : Convert Python's [AST](https://docs.Python.org/3/library/ast.html) (abstract syntax tree) representation of the Python file to Pyccel's AST representation (objects of the classes in the folder [pyccel.ast](../pyccel/ast))
2.  [**Errors**](#Errors) : Raise an error for any syntax used that is not yet supported by pyccel
3.  [**Headers**](#Headers) : Convert header comments from strings to Pyccel's AST representation
4.  [**Scoping**](#Scoping) : Collect the name of all variables in each scope (see [scope](scope.md) for more details) to ensure no name collisions can occur if pyccel generates Variable names

## Navigation and AST Creation

The entry point for the class `SyntaxParser` is the function `parse`.
This function is called from the constructor to examine a FST (Full Syntax Tree) object.
The FST is the output of Python's `ast.parse` function with comments inserted using the [`pyccel.parser.extend_tree.extend_tree`](../pyccel/parser/extend_tree.py).
Python's ast module discards all comments, so the function [`pyccel.parser.extend_tree.extend_tree`](../pyccel/parser/extend_tree.py) is needed to read the file and reinsert these comments at the correct location.

The key line of the function `parse` is the call to `self._visit(self.fst)`.
All elements of the tree must be visited.
The `_visit` function internally calls a function named `_visit_X`, where `X` is the type of the object.
These functions must have the form:
```python
def _visit_ClassName(self, stmt):
    ...
```
Each of these `_visit_X` functions should internally call the `_visit` function on each of the elements of the object to obtain Pyccel AST nodes which can be combined to create a Pyccel AST node representing the current object.

### Example

Consider for example the Python expression:
```python
a += b
```
The Python ast module parses this as:
```python
AugAssign(target=Name(id='a', ctx=Store()), op=Add(), value=Name(id='b', ctx=Load()))
```
and we want to save this information into Pyccel's [`pyccel.ast.core.AugAssign`](../pyccel/ast/core.py).
We therefore need to visit the 2 members of `ast.AugAssign`:
-   target
-   value

and use the third (`op`) to correctly construct the `pyccel.ast.core.AugAssign`.

The final code therefore ressembles the following:
```python
def _visit_AugAssign(self, stmt):
    lhs = self._visit(stmt.target)
    rhs = self._visit(stmt.value)
    if isinstance(stmt.op, ast.Add):
        return AugAssign(lhs, '+', rhs)
    elif ...:
        ...
    else:
        return errors.report(PYCCEL_RESTRICTION_TODO, symbol = stmt,
                severity='error')
```

## Errors

Error handling uses the classes in the file [pyccel.errors.errors](../pyccel/errors/errors.py).
Errors in the syntactic stage should raise a `PyccelSyntaxError`.
Where possible this should be done by accessing the `Errors()` singleton and calling the `report` function.
This function takes several arguments (see docstring for more details).
The most important arguments are:
-   _message_ : Describe the issue that lead to the error

-   _symbol_ : The Python ast object should be passed here. This object contains information about its position in the file (line number, column) which ensures the user can more easily locate their error

-   _severity_ : The severity level must be one of the following:
    -   _warning_ : An error will be printed but Pyccel will continue executing
    -   _error_ : An error will be printed but Pyccel will continue executing the syntactic stage
    -   _fatal_ : An error will be printed and Pyccel will stop executing. This level should rarely be needed in the syntactic stage as a failure in one visitation function (`_visit_X`) should not affect the execution of another. It is preferable to show the users all errors at once

## Headers

The headers (type declarations/openmp pragmas/etc) also have their own syntax which cannot be parsed by Python's ast module.
The module [textx](http://textx.github.io/textX/stable/) is used to parse these statements.
The files describing the _textx_ grammar are found in the folder [pyccel.parser.grammar](../pyccel/parser/grammar).
From these files _textx_ generates instances of the classes found in the folder [pyccel.parser.syntax](../pyccel/parser/syntax).
These instances can then be inserted into the abstract syntax tree.

## Scoping

The final purpose of the syntactic stage is to collect all names declared in the code.
This is important to avoid name collisions if pyccel creates temporaries or requires additional names.
The names are saved in the scope (for more details see [scope](scope.md)).
Whenever a symbol is encountered in a declaration context, it should be saved to the scope using the function `self.scope.insert_symbol`.
This is usually done in the `_visit_Name` function, however this function is not aware of the context so it cannot determine whether it is a declaration.
To get round this the class has the property `SyntaxParser._in_lhs_assign`, which should be `True` in a declaration context, and false elsewhere.

Consider for example a for loop. Such a loop has 3 main parts (which are each members of `ast.For`):
-   target  (`ast.For.target`)
-   iterable  (`ast.For.iter`)
-   body (`ast.For.body`)

such that a for loop is defined as:
```python
for target in iterable:
   body
```

Each of these 3 members must be visited individually, but the target additionally is in a declaration context.
The visitation section of `_visit_For` is therefore:
```python
self._in_lhs_assign = True
iterator = self._visit(stmt.target)
self._in_lhs_assign = False
iterable = self._visit(stmt.iter)
body = self._visit(stmt.body)
```

### Scoped Node
Any functions visiting a class which inherits from `ScopedNode` must create a new scope before visiting objects and exit it after everything inside the scope has been visited.
The scope must then be passed to the class using the keyword argument `scope`.
Care should be taken here as this keyword is not compulsory[^1].

[^1]: The keyword cannot currently be compulsory due to some old code and the special case of `FunctionDef`. A `FunctionDef` which represents a function defined in a header file has no body, and therefore no scope.

A child scope can be created using one of the following functions (for more details see the docstrings in [pyccel.parser.scope](../pyccel/parser/scope.py):
-   `Scope.new_child_scope`
-   `Scope.create_new_loop_scope`
-   `Scope.create_product_loop_scope`

### Temporary Creation
Occasionally it is necessary to create objects in the syntactic stage.
The `Scope` functions should be used for this purpose to avoid name collisions.
See the [scope](scope.md) docs for more details.
In all cases it is preferable to delay this stage as long as possible to ensure that as much information is known about the scope as possible.
This is important as at this stage there may still be conflicting names which appear later in the file.
The `Scope` should prevent name collisions with these objects, but that will lead to them being renamed which makes the translated code harder to recognise when compared with the original.

## History

Originally the syntactic stage translated from [RedBaron](https://github.com/PyCQA/redbaron)'s AST representation to Pyccel's AST representation.
RedBaron parses Python, but makes no attempt to validate the code.
This made pyccel's job harder as there was no guarantee that the syntax was correct.
Since moving to Python's ast module the syntactic stage has been massively simplified.
This is because Python's ast module checks the validity of the syntax.
