# Change Log
All notable changes to this project will be documented in this file.

## \[Unreleased\]

### Added

-   Allow interfaces in classes.
-   Python support for a simple class.
-   #1430 : Added conjugate support to booleans.
-   #1452 : Added C support for a class containing only functions

### Fixed

-   \[INTERNALS\] Fix string base class selection.

### Changed

-   Updating `stdlib` files if they are modified not just accessed.
-   \[INTERNALS\] Add class object to class function call arguments.

### Deprecated

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
