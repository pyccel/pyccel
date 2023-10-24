# Change Log
All notable changes to this project will be documented in this file.

## \[1.10.0\] - 2023-10-23

### Added

-   #633 & #1518 : Allow non-trivial types to be specified with mypy-compatible annotations.
-   #1336 : Use template as a partial type.
-   #1509 : Add type annotations for variables.
-   #1528 : Add support for variable declarations in classes.

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

-   Removed support for untested, undocumented `lambidify` function.
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
-   \[INTERNALS\] Remove unused `Declare.dtype`.
-   \[INTERNALS\] Remove unused functions `subs`, `inline`, `get_iterable_ranges` from `pyccel.ast.core`.
-   \[INTERNALS\] Remove unused class `pyccel.ast.core.ForIterator`.
-   \[INTERNALS\] Remove unused parameters `expr`, `status` and `like` from `pyccel.ast.core.Assign`.

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
-   #929 : Allow optional variables when compiling with intel or nvidia.
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
