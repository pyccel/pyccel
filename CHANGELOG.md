# Change Log
All notable changes to this project will be documented in this file.

## \[Unreleased\]

### Added
-   #1396 : Added the `--ignore-conda-warnings` and `--detailed-conda-warnings`, and updating the `compiler.md`.
### Fixed

### Changed

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

### Deprecated

## \[1.7.3\] - 2023-03-07

### Added

-   Improved developer docs (code generation stage).

### Fixed

-   #1337 : Bug causing overflow errors when templates are used in functions with a large number of arguments.
-   #892 : Bug in the wrapper preventing an argument from using templates to have both a scalar and an array type.

### Changed

-   \[INTERNALS\] Add validation of docstrings to CI.

### Deprecated

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

### Deprecated

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

### Deprecated
