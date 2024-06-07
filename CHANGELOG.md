# Change Log
All notable changes to this project will be documented in this file.

## \[UNRELEASED\]

### Added

-   #1720 : Add support for `Ellipsis` as the only index for an array.
-   #1694 : Add Python support for list method `extend()`.
-   #1700 : Add Python support for list method `sort()`.
-   #1696 : Add Python support for list method `copy()`.
-   #1693 : Add Python support for list method `remove()`.
-   #1739 : Add abstract class `SetMethod` to handle calls to various set methods.
-   #1739 : Add Python support for set method `clear()`.
-   #1740 : Add Python support for set method `copy()`.
-   #1750 : Add Python support for set method `remove()`.
-   #1743 : Add Python support for set method `discard()`.
-   #1754 : Add Python support for set method `update()`.
-   #1787 : Ensure `STC` is installed with Pyccel.
-   #1656 : Ensure `gFTL` is installed with Pyccel.
-   #1844 : Add line numbers and code to errors from built-in function calls.
-   #1655 : Add the appropriate C language equivalent for declaring a Python `list` container using the `STC` library.
-   #1659 : Add the appropriate C language equivalent for declaring a Python `set` container using the `STC` library. 
-   \[INTERNALS\] Added `container_rank` property to `ast.datatypes.PyccelType` objects.
-   \[DEVELOPER\] Added an improved traceback to the developer-mode errors for errors in function calls.
-   #1893 : Add Python support for set initialisation with `set()`.

### Fixed

-   #1720 : Fix Undefined Variable error when the function definition is after the variable declaration.
-   #1763 Use `np.result_type` to avoid mistakes in non-trivial NumPy type promotion rules.
-   Fix some cases where a Python built-in type is returned in place of a NumPy type.
-   Stop printing numbers with more decimal digits than their precision.
-   Allow printing the result of a function returning multiple objects of different types.
-   #1732 : Fix multidimensional list indexing in Python.
-   #1785 : Add missing cast when creating an array of booleans from non-boolean values.
-   #1821 : Ensure an error is raised when creating an ambiguous interface.
-   #1842 : Fix homogeneous tuples incorrectly identified as inhomogeneous.
-   #1853 : Fix translation of a file whose name conflicts with Fortran keywords.
-   Link and mention `devel` branch, not `master`.
-   #1047 : Print the value of an unrecognised constant.
-   #1903 : Fix memory leak when using type annotations on local variables.

### Changed

-   #1836 : Move `epyccel` module to `pyccel.commands.epyccel` and add support for shortcut import `from pyccel import epyccel`.
-   #1720 : functions with the `@inline` decorator are no longer exposed to Python in the shared library.
-   #1720 : Error raised when incompatible arguments are passed to an `inlined` function is now fatal.
-   \[INTERNALS\] `FunctionDef` is annotated when it is called, or at the end of the `CodeBlock` if it is never called.
-   \[INTERNALS\] `InlinedFunctionDef` is only annotated if it is called.
-   \[INTERNALS\] Build `utilities.metaclasses.ArgumentSingleton` on the fly to ensure correct docstrings.
-   \[INTERNALS\] Rewrite datatyping system. See #1722.
-   \[INTERNALS\] Moved precision from `ast.basic.TypedAstNode` to an internal property of `ast.datatypes.FixedSizeNumericType` objects.
-   \[INTERNALS\] Moved rank from `ast.basic.TypedAstNode` to an internal property of `ast.datatypes.PyccelType` objects.
-   \[INTERNALS\] Moved order from `ast.basic.TypedAstNode` to an internal property of `ast.datatypes.PyccelType` objects.
-   \[INTERNALS\] Use cached `__add__` method to determine result type of arithmetic operations.
-   \[INTERNALS\] Use cached `__and__` method to determine result type of bitwise comparison operations.
-   \[INTERNALS\] Stop storing `FunctionDef`, `ClassDef`, and `Import` objects inside `CodeBlock`s.
-   \[INTERNALS\] Remove the `order` argument from the `pyccel.ast.core.Allocate` constructor.
-   \[INTERNALS\] Remove `rank` and `order` arguments from `pyccel.ast.variable.Variable` constructor.
-   \[INTERNALS\] Ensure `SemanticParser.infer_type` returns all documented information.
-   \[INTERNALS\] Enforce correct value for `pyccel_staging` property of `PyccelAstNode`.
-   \[INTERNALS\] Allow visiting objects containing both syntactic and semantic elements in `SemanticParser`.
-   \[INTERNALS\] Rename `pyccel.ast.internals.PyccelInternalFunction` to `pyccel.ast.internals.PyccelFunction`.
-   \[INTERNALS\] All internal classes which can be generated from `FunctionCall`s must inherit from `PyccelFunction`.
-   \[INTERNALS\] `PyccelFunction` objects which do not represent objects in memory have the type `SymbolicType`.
-   \[INTERNALS\] Rename `_visit` functions called from a `FunctionCall` which don't match the documented naming pattern to `_build` functions.
-   \[INTERNALS\] Remove unnecessary argument `kind` to `Errors.set_target`.

### Deprecated

-   #1786 : Remove support for `real` and `integer` as type annotations.
-   #1812 : Stop allowing multiple main blocks inside a module.
-   \[INTERNALS\] Remove property `ast.basic.TypedAstNode.precision`.
-   \[INTERNALS\] Remove class `ast.datatypes.DataType` (replaced by `ast.datatypes.PrimitiveType` and `ast.datatypes.PyccelType`).
-   \[INTERNALS\] Remove unused properties `prefix` and `alias` from `CustomDataType`.
-   \[INTERNALS\] Remove `ast.basic.TypedAstNode._dtype`. The datatype can still be accessed as it is contained within the class type.
-   \[INTERNALS\] Remove unused parameters `expr`, `status` and `like` from `pyccel.ast.core.Assign`.
-   \[INTERNALS\] Remove `pyccel.ast.utilities.builtin_functions`.
-   \[INTERNALS\] Remove unused/unnecessary functions in `pyccel.parser.utilities` : `read_file`, `header_statement`, `accelerator_statement`, `get_module_name`, `view_tree`.
-   \[INTERNALS\] Remove unused functions `Errors.unset_target`, and `Errors.reset_target`.

## \[1.12.0\] - 2024-05-08

### Added

-   #1830 : Add a `pyccel.lambdify` function to accelerate SymPy expressions.
-   #1867 : Add a `use_out` parameter to `pyccel.lambdify` to avoid unnecessary memory allocation.
-   #1867 : Auto-generate a docstring for functions generated via calls to `pyccel.lambdify`.
-   #1868 : Hide traceback for `epyccel` and `lambdify` errors.

### Fixed

-   #1762 : Fix array copy between different data types.
-   #1792 : Fix array unpacking.
-   #1795 : Fix bug when returning slices in C.
-   #1218 : Fix bug when assigning an array to a slice in Fortran.
-   #1830 : Fix missing allocation when returning an annotated array expression.
-   #1853 : Fix translation of a file whose name conflicts with Fortran keywords.
-   Link and mention `devel` branch, not `master`.

### Changed

-   #1866 : Raise a more informative error when mixing scalar and array return types.
-   \[TESTS\] Filter out cast warnings in cast tests.
-   \[INTERNALS\] Removed unused `fcode`, `ccode`, `cwrappercode`, `luacode`, and `pycode` functions from printers.
-   \[INTERNALS\] Removed unused arguments from methods in `pyccel.codegen.codegen.Codegen`.

### Deprecated

-   #1820 : Deprecated unused decorator `@lambdify`
-   \[INTERNALS\] Removed unused and undocumented function `get_function_from_ast`.
-   \[INTERNALS\] Remove function `Module.set_name`.
-   \[INTERNALS\] Remove unused `assign_to` argument of `CodePrinter.doprint`.

## \[1.11.2\] - 2024-03-05

### Added

-   #1689 : Add Python support for list method `append()`.
-   #1692 : Add Python support for list method `insert()`.
-   #1690 : Add Python support for list method `pop()`.
-   #1691 : Add Python support for list method `clear()`.
-   #1575 : Add support for homogeneous tuple type annotations on variables.
-   #1425 : Add support for `numpy.isnan`, `numpy.isinf` and `numpy.isfinite`.
-   #1738 : Add Python support for creating scalar sets with `{}`.
-   #1738 : Add Python support for set method `add`.
-   #1749 : Add Python support for set method `pop()`.

### Fixed

-   #1575 : Fixed inhomogeneous tuple (due to incompatible sizes) being treated as homogeneous tuple.
-   #1182 : Fix tuples containing objects with different ranks.
-   #1575 : Fix duplication operator for non-homogeneous tuples with a non-literal but constant multiplier.
-   #1779 : Fix standalone partial templates.

### Changed

-   #1776 : Increase minimum version for `pytest` to 7.0.

### Deprecated

-   \[INTERNALS\] Remove unnecessary `dtype` parameter from `ast.core.Declare` class.
-   \[INTERNALS\] Remove unnecessary `passed_from_dotted` parameter from `ast.core.Declare` class.
-   \[INTERNALS\] Remove unused `ast.core.Block` class.

## \[1.11.1\] - 2024-02-13

### Fixed

-   #1724 : Fix returns in for loops

## \[1.11.0\] - 2024-02-12

### Added

-   #1645 : Handle deprecated `ast` classes.
-   #1649 : Add support for `np.min` in C code.
-   #1621 : Add support for `np.max` in C code.
-   #1571 : Add support for the function `tuple`.
-   #1493 : Add preliminary support for importing classes.
-   #1578 : Allow classes to avoid type annotations for the self argument of a method.
-   #1597 : Handle class docstrings.
-   #1494 : Add support for functions returning class instances.
-   #1495 : Add support for functions with class instance arguments.
-   #1684 : Add support for classes without `__init__` functions.
-   #1685 : Add support for `type()` function with class instance argument.
-   #1605 : Add support for methods and interfaces in classes (including `__init__` and `__del__`).
-   #1618 : Add support for class instance attributes.
-   #1680 : Add support for `typing.Final`.
-   Add a `--time_execution` flag to allow detailed investigation of critical sections of code.
-   #1659 : Add multi-file support for classes.
-   #1708 : Allow returning pointers to arguments from functions.
-   \[INTERNALS\] Add `class_type` attribute to `TypedAstNode`.
-   \[INTERNALS\] Add `PyccelPyArrayObject` datatype.

### Fixed

-   #1587 : Fix unnecessarily long file names generated by `epyccel`.
-   #1576 : Correct destructor invocation for proper cleanup.
-   #1576 : Remove inline class method definition.
-   Ensure an error is raised when if conditions are used in comprehension statements.
-   #1553 : Fix `np.sign` when using the `ifort` compiler.
-   #1582 : Allow homogeneous tuples in classes.
-   #1619 : Give priority to imported functions over builtin functions.
-   #1614 : Allow relative paths for custom compilation file.
-   #1615 : Fixed infinite loop when passing slices while copying arrays.
-   #1628 : Fixed segmentation fault when writing to optional scalars.
-   #1554 : Fix exit statement in Fortran with Intel compiler.
-   #1564 : Fixed installation problems on Python 3.12.
-   #1259 : Fix bug causing problems with user editable installation.
-   #1651 : Fix name collision resolution to include parent scopes.
-   #1156 : Raise an error for variable name collisions with non-variable objects.
-   #1507 : Fix problems with name collisions in class functions.
-   Ensure `pyccel-init` calls the related function.
-   Stop unnecessarily importing deprecated NumPy classes `int`, `bool`, `float`, `complex` in Python translation.
-   #1712 : Fix library path and OpenMP support for recent Apple chips by getting Homebrew directory with `brew --prefix`.
-   #1687 : Pointers in tuples are deallocated.
-   #1586 : Raise an error for targets of class instances which go out of scope too early.
-   #1717 : Fix a bug when handling paths with dots.

### Changed

-   #1672 : Make `icx` and `ifx` the default Intel compilers (Found in Intel oneAPI).
-   #1644 : Stop printing the step of a range if that step is 1.
-   #1638 : Migrate from `setuptools` to `hatch` for installation scripts.
-   Don't raise a warning for an unnecessary specification of the order.
-   \[INTERNALS\] #1593 : Rename `PyccelAstNode.fst` to the `PyccelAstNode.ast`.
-   \[INTERNALS\] #1593 : Use a setter instead of a method to update `PyccelAstNode.ast`.
-   \[INTERNALS\] #1593 : Rename `BasicParser._current_fst_node` to the `BasicParser._current_ast_node`.
-   \[INTERNALS\] #1390 : Remove dead code handling a `CodeBlock` in an assignment.
-   \[INTERNALS\] #1582 : Remove the `HomogeneousTupleVariable` type.
-   \[INTERNALS\] #1581 : Unify handling of string and Python annotations.

### Deprecated

-   #1593 : Remove undocumented, broken `lambdify` method.
-   \[INTERNALS\] #1584 : Remove unused functions from `pyccel.ast.core` : `inline`, `subs`, `get_iterable_ranges`.
-   \[INTERNALS\] #1584 : Remove unused functions from `pyccel.ast.datatypes` : `is_iterable_datatype`, `is_with_construct_datatype`, `is_pyccel_datatype`.
-   \[INTERNALS\] #1584 : Remove unused class from `pyccel.ast.core`: `ForIterator`.
-   \[INTERNALS\] #1584 : Remove unused method from `pyccel.ast.core`: `ClassDef.get_attribute`.
-   \[INTERNALS\] #1676 : Remove `DottedFunctionCall` from `pyccel.ast.core` (use `bound_argument` instead).
-   \[INTERNALS\] #1683 : Remove unused redundant class from `pyccel.ast.datatypes`: `UnionType`.

## \[1.10.0\] - 2023-10-23

### Added

-   #633 & #1518 : Allow non-trivial types to be specified with mypy-compatible annotations.
-   #1336 : Use template as a partial type.
-   #1509 : Add type annotations for variables.
-   #1528 : Add support for variable declarations in classes.
-   #1491 : Add documentation for classes.

### Fixed

-   #387 : Raise a clear error when an unrecognised type is used in a type annotation.
-   #1556 : Fixed print format string for Intel compatibility.
-   #1557 : Fix return a new instance of a class.
-   #1557 : Fix save multiple class instances to the same variable.

### Changed

-   \[INTERNALS\] #1520 : `ScopedNode` -> `ScopedAstNode`.
-   \[INTERNALS\] #1520 : `PyccelAstNode` -> `TypedAstNode`.
-   \[INTERNALS\] #1520 : `Basic` -> `PyccelAstNode`.

### Deprecated

-   Drop official support for Python 3.7 due to End of Life.

## \[1.9.2\] - 2023-10-13

### Added

-   #1476 : Add C support for a class containing `Interfaces`.
-   #1472 : Add C printing support for a class containing scalar data.
-   #1492 : Types of arguments for class methods can be declared like function arguments.
-   #1511 : Add support for the `cmath` library.
-   Output readable header syntax errors.
-   New environment variable `PYCCEL_DEFAULT_COMPILER`.
-   #1508 : Add C support for a class destructor.
-   #1508 : Add support for array data in classes.

### Fixed

-   #1484 : Use scope for classes to avoid name clashes.
-   Stop raising warning for unrecognised functions imported via intermediate modules.
-   #1156 : Raise a neat error for unhandled inhomogeneous tuple expressions.
-   Set status of header variables to 'unallocated'.
-   #1508 : Generate deallocations for classes and their attributes.

### Changed

-   #1484 : Improve handling of `DottedName` in `_assign_lhs_variable`.
-   \[INTERNALS\] Move handling of variable headers to semantic stage.
-   \[INTERNALS\] Moved handling of type annotations to the semantic stage.
-   \[INTERNALS\] Remove unnecessary body argument from `FunctionAddress`.

### Deprecated

-   #1513 : Stop printing `@types` decorators in generated Python code.
-   Remove support for undocumented type syntax specifying precision (e.g. `int*8`).
-   No longer possible to combine header annotations and argument type annotations.
-   Remove support for specifying header annotations in a separate file.
-   \[INTERNALS\] Remove `dtype_registry` in favour of `dtype_and_precision_registry`.
-   \[INTERNALS\] Prefer `DataType` keys over string keys which describe data types.

## \[1.9.1\] - 2023-08-31

### Added

-   #1497 : Add support for NumPy `copy` method: `a.copy`.
-   #1497 : Add support for NumPy function `copy`.

### Fixed

-   #1499 : Fix passing temporary arrays to functions.
-   #1241 : Missing transpose when converting from a C-ordered array to F-ordered array.
-   #1241 : Incorrect transpose when copying an F-ordered array.
-   #1241 : Fix infinite loop when passing an array as the only argument to `np.array`.
-   #1506 : Increment `Py_None` reference count to avoid unexpected deallocation.

## \[1.9.0\] - 2023-08-22

### Added

-   #752 : Allow passing array variables to `numpy.array`.
-   #1280 : Allow copying arrays using `numpy.array`.
-   Allow interfaces in classes.
-   Add Python support for a simple class.
-   #1430 : Add conjugate support to booleans.
-   #1452 : Add C printing support for a class containing only functions.
-   #1260 : Add support for NumPy `dtype` property: `a.dtype`.
-   #1260 : Add support for NumPy `result_type` function.

### Fixed

-   #682 : Wrong data layout when copying a slice of an array.
-   #1453 : Fix error-level developer mode output.
-   \[INTERNALS\] Fix string base class selection.
-   #1496 : Fix interfaces which differ only by order or rank.

### Changed

-   #1455 : Make `ConstructorCall` inherit from `FunctionCall`.
-   Updating `stdlib` files if they are modified not just accessed.
-   `pyccel_clean` tool now deletes folders **starting with** `__pyccel__` and `__epyccel__`.
-   Pyccel-generated folder names are dependent on `PYTEST_XDIST_WORKER` when running with `pytest-xdist`.
-   \[INTERNALS\] Add class object to class function call arguments.
-   \[INTERNALS\] In `ast.numpyext` rename `Shape` as `NumpyShape`, `NumpyArraySize` as `NumpySize`
-   \[INTERNALS\] In `ast.internals` rename `PyccelArraySize` as `PyccelArraySizeElement`, create new `PyccelArraySize` w/out `index` argument
-   \[INTERNALS\] Make `NumpySize` a factory class (which cannot be instantiated)
-   \[INTERNALS\] Re-write C-Python API wrapping stage (#1477)

### Deprecated

-   Using a `@types` decorator will raise a `FutureWarning` as this will be deprecated in a future version.
-   Using a type specification header will raise a `FutureWarning` as this will be deprecated in a future version.
-   Stop generating `numpy.bool` (deprecated from NumPy) in code.
-   \[INTERNALS\] Removed `obsolete` folder.
-   \[INTERNALS\] Removed out of date `samples` folder.
-   \[INTERNALS\] Removed out of date `doc` folder.
-   \[INTERNALS\] Removed `benchmarks` folder. Code is still available in benchmark repository.
-   \[INTERNALS\] Removed `bugs` folder.
-   \[INTERNALS\] Removed `inprogress` folder.
-   \[INTERNALS\] Remove old Travis configuration file.

## \[1.8.1\] - 2023-07-07

### Added

-   #1430 : Added conjugate support to integers and floats.

### Fixed

-   #1427 : Fix augmented assignment with a literal right hand side in templated code.

## \[1.8.0\] - 2023-06-20

### Added
-   #1400 : Added flags to Pyccel for managing conda PATH warnings.

### Fixed

-   #1404 : Bug preventing printing of some functions in a `print()` call.
-   #1302 : Raise error message in case of empty class.
-   #1407 : Raise an error if file name matches a Python built-in module.
-   #929 : Allow optional variables when compiling with Intel or NVIDIA.
-   #1117 : Allow non-contiguous arrays to be passed to Fortran code.
-   #1415 : Fix incorrect handling of assignments augmented by function calls.
-   #1418 : Fix `itertools.product` implementation.

### Changed

-   #1355 : Remove unused `BasicParser` arguments.
-   \[INTERNALS\] Re-write bind-c wrapping stage (#1388)

## \[1.7.4\] - 2023-05-02

### Added

-   #1352 : Added a change log.

### Fixed

-   #1367 : Use an absolute path to link to Python library.
-   #1379 : Ensure temporary arrays created for function calls are correctly declared in loops.

### Changed

-   Default to linking Python dynamically instead of statically
-   Ensure only absolute paths are used in compilation command.
-   \[INTERNALS\] Use `FunctionDefArgument` to store all argument specific properties.
-   \[INTERNALS\] Reduce carbon footprint by avoiding unnecessary CI testing.
-   \[INTERNALS\] Automatise PR labelling and review progress prompts.
-   \[INTERNALS\] Enforce the use of `FunctionDefArgument` in `FunctionDef`.
-   \[INTERNALS\] Use `FunctionDefResult` to store all result specific properties.

## \[1.7.3\] - 2023-03-07

### Added

-   Improved developer docs (code generation stage).

### Fixed

-   #1337 : Bug causing overflow errors when templates are used in functions with a large number of arguments.
-   #892 : Bug in the wrapper preventing an argument from using templates to have both a scalar and an array type.

### Changed

-   \[INTERNALS\] Add validation of docstrings to CI.

## \[1.7.2\] - 2023-02-02

### Added

### Fixed

-   #1288 : Bug in slice indexing in C code.
-   #1318 : Bug preventing use of `np.linspace` more than once in a given function.

### Changed

-   \[INTERNALS\] Uniformise line endings and enforce the convention through the use of a `.gitattributes` file.
-   \[INTERNALS\] Add human-readable summaries to tests.
-   \[INTERNALS\] Add tests to ensure Pyccel conventions are followed.
-   \[INTERNALS\] Add tests to check spelling.

## \[1.7.1\] - 2023-01-26

### Added

-   #1309 : Support for `np.sum` in C code.
-   Improved [developer docs](./developer_docs) (ordering, syntactic stage, semantic stage).
-   Added [community guidelines](./github/CONTRIBUTING.md).

### Fixed

-   #1184 : Bug preventing compilation on systems where there is no static library available for Python.
-   #1281 : Bug causing assignment to pointer instead of incrementation.
-   #1282 : Imported constants cannot be returned from functions.
-   \[INTERNALS\] Bug in CI coverage for forks.

### Changed

-   #1315 : Installation process modified to make test requirements a pip optional dependency.
-   #1245 : Reduce false negative test results by using a tolerance to compare floats.
-   #1272 : Remove use of deprecated NumPy syntax in tests.
-   #1253 : Provide minimum requirements.
-   \[INTERNALS\]  #1385 : Remove unused settings keyword arguments from `_visit` function.
