## Developer Setup

Before beginning any development in Pyccel, it is important to ensure Pyccel is correctly installed **from source in development mode** as described [here](../README.md#from-sources). If this step is not followed then any changes made to source will not be used when `pyccel` or `epyccel` are used.

## Overview

Pyccel's development is split into 4 main stages:

-   [Syntactic Stage](#Syntactic-stage) (for more details see [syntactic stage](syntactic_stage.md))
-   [Semantic Stage](#Semantic-stage) (for more details see [semantic stage](semantic_stage.md))
-   [Code Generation Stage](#Code-generation-stage) (for more details see [codegen stage](codegen_stage.md))
-   [Wrapping Stage](#Wrapping-stage) (for more details see [wrapping stage](wrapping_stage.md))
-   [Compilation Stage](#Compilation-stage)

### Syntactic Stage

Pyccel uses Python's [`ast` module](https://docs.python.org/3/library/ast.html) to read the input file(s). The abstract syntax tree (AST) of Python's `ast` module does not store information in the same way as the rest of Pyccel so this stage exists to **convert Python's AST to Pyccel's AST**. The related code can be found in [parser/syntactic.py](../pyccel/parser/syntactic.py).

The syntactic stage also handles parsing header comments. This is managed using [textx](http://textx.github.io/textX/stable/). The files describing the _textx_ grammar are found in the folder [parser/grammar](../pyccel/parser/grammar). From these files _textx_ generates instances of the classes found in the folder [parser/syntax](../pyccel/parser/syntax).

#### Advanced Comments

The role of this stage has decreased significantly since we moved from [redbaron](https://redbaron.readthedocs.io/en/latest/) to Python's [ast module](https://docs.python.org/3/library/ast.html). At some point in the future it may therefore be worth asking whether this stage is still pertinent.

### Semantic Stage

This is the most important stage in Pyccel. It is here that all the information about types is calculated. This stage strives to be **language-agnostic**; this means for example, that additional variables required to handle problems appearing in one specific language should not be created here.

When adding functions to this stage the aim is often to create a `PyccelAstNode` (see [ast/basic.py](../pyccel/ast/basic.py)) and correctly define all of its parameters. This information is sometimes readily available (e.g. the type of a `PyccelAdd` can be derived from the type of the variables passed to it), but sometimes the information must be collected from elsewhere (e.g. when creating a `Variable` from a `PyccelSymbol`, which is roughly equivalent to a string). In this case information is needed from a `Scope` instance which is stored in the `scope`.

In computer science, the _scope_ is the area of a program where an item (e.g. variable, function, etc.) is recognised. For example a variable defined in a function will not be recognised outside of that function, therefore the function defines its scope.

In Pyccel, a `Scope` is an object defined in [parser/base.py](../pyccel/parser/base.py) which represents this concept. It includes all the functions, imports, variables, and classes which are available at a given point in the code. It also contains pointers to nested and parent scopes. The `scope` in the `SemanticParser` (`SemanticParser._scope`) stores the Scope relevant to the line of code being treated. It must be updated whenever the scope changes (e.g. through the `create_new_function_scope` function when entering into a function body). The Scope is also used to avoid naming collisions. See [Scope documentation](./scope.md) for details.

### Code Generation Stage

In this stage the Pyccel nodes are converted into a string which contains the translation into the requested language. Each language has its own printer. The printers are found in the folder [codegen/printing](../pyccel/codegen/printing).

As in the Semantic stage, the Code Generation stage also stores the current Scope in a `scope` variable which must be updated whenever the scope changes.

### Wrapping Stage

In this stage, Pyccel creates a wrapper to interface the generated low-level code with Python. The wrapper is a C file which uses the [Python/C API](https://docs.python.org/3/c-api/intro.html) and the [NumPy C-API](https://numpy.org/doc/stable/reference/c-api/index.html).

### Compilation Stage

Finally the generated code is compiled. This is handled in the [pipeline](../pyccel/codegen/pipeline.py). The compilers commands are found in [codegen/compiling/compilers.py](../pyccel/codegen/compiling/compilers.py). Different compilers have different flags and need different libraries. Once Pyccel has been executed once on your machine the flags and libraries can be found in JSON files in the [compilers](../pyccel/compilers) folder.

### Function Naming Conventions/File Navigation

In the syntactic, semantic, and code generation stages a similar strategy is used for traversing the Python objects. This strategy is based on function names. The majority of functions have names of the form: `_prefix_ClassName` (in the syntactic and semantic stages the prefix is `visit`, in the code generation stages it is `print`). These functions are never called directly, but instead are called via a high level function `_prefix` (e.g. `_visit` for the semantic stage). This strategy avoids large `if`/`elif` blocks to handle all possible types.

#### Example
Suppose we want to generate the code for an object of the class `NumpyTanh`, first we collect the inheritance tree of `NumpyTanh`. This gives us:
```python
('NumpyTanh', 'NumpyUfuncUnary', 'NumpyUfuncBase', 'PyccelInternalFunction', 'PyccelAstNode', 'Basic')
```
Therefore the print functions which are acceptable for visiting this object are:

-   `_print_NumpyTanh` 
-   `_print_NumpyUfuncUnary` 
-   `_print_NumpyUfuncBase` 
-   `_print_PyccelInternalFunction` 
-   `_print_PyccelAstNode` 
-   `_print_Basic` 

We run through these possible functions choosing the one which is the most specialised. If none of these methods exist, then an error is raised.

In the case of `NumpyTanh` the function which will be selected is `_print_NumpyUfuncBase` when translating to C or Fortran, and `_print_PyccelInternalFunction` when translating to Python.

### AST

The objects as understood by Pyccel are each described by classes which inherit from [pyccel.ast.basic.Basic](../pyccel/ast/basic.py).
These classes are found in the [ast](../pyccel/ast) folder.
The objects in the Abstract Syntax Tree (AST) are described in several files.
There is one file for each supported extension module and files to group concepts, e.g. literals/operators/built-in functions.

## Error System

Pyccel tries to fail cleanly and raise readable errors for users. This is managed using the error handling module found in the [errors](../pyccel/errors) folder. In order to raise an error 2 things must be done:

1.  An instance of the singleton class `Errors` must be created
2.  The `report` method of the class must be called (see docstring for details)

If the error prevents further translation (e.g. the type of an object is now unknown) then the severity should be indicated as `'fatal'`.

## Getting Help

While discussions within the associated GitHub issue are often sufficient, should you require more help you can join our [Pyccel Discord Server](https://discord.gg/2Q6hwjfFVb).
