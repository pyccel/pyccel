# Interfacing Python code with low-level code

_Introduced in Pyccel v2.1._

Pyccel now supports describing libraries written in low-level languages, directly using small `.pyi` interface files. These files act as a contract: they declare the available functions, methods, and types, along with precise low-level signatures.

This gives you two benefits at once:

- The `pyccel-wrap` command allows you to call the compiled code directly from Python.

- Pyccel understands those calls, so when you translate Python to C/Fortran code using Pyccel, the calls are re-emitted as direct calls to the original routines.

In practice, the `.pyi` file creates a two-way bridge between Python and low-level code: easy to call from Python, and still translatable by Pyccel.

## Contents

- [Typical workflow (at a glance)](#typical-workflow-at-a-glance)
- [Writing the stub file](#writing-the-stub-file)
    - [Name mapping with `@low_level`](#name-mapping-with-low_level)
    - [Compilation metadata](#compilation-metadata)
    - [Fortran specific rules](#fortran-specific-rules)
    - [C specific rules](#c-specific-rules)
- [Fortran Example](#fortran-example)
- [C Example](#c-example)

## Typical workflow (at a glance)

- Write/compile your C/Fortran library.

- Write a `.pyi` stub with precise types and minimal build metadata.

- Wrap the library with a Python interface using `pyccel-wrap` to make it callable from Python.

- Develop in Python using the interface to the low-level code.

- Run Pyccel to translate the new Python code back to C/Fortran code.

This preserves a clean Python API while enabling Pyccel to treat your calls as first-class, translatable operations.

## Writing the stub file

The stub file (`.pyi`) describes the low-level library in a way that is understandable in a Python environment. It follows the same conventions as standard Python stub files (see the [Python documentation](https://typing.python.org/en/latest/guides/writing_stubs.html)), with a few additional rules specific to Pyccel. These additional rules mean that Pyccel cannot create an interface for every C/Fortran function.

In addition to the general rules above there are more specific rules for [Fortran](#fortran-specific-rules) and [C](#c-specific-rules) which are detailed later.
The following are the general language-agnostic rules:

- At the top of the file, you must include some metadata about the compilation

- If you are exposing a class to Python, the stub must define `__init__` and `__del__` methods, which map to the corresponding routines in the low-level code.

You must ensure that all functions, methods, and classes have precise type annotations that match the low-level signatures.
Particular care should be taken with integers. On most platforms Python's default integer precision is equivalent to `numpy.int64`, while the default integer precision in low-level languages like C and Fortran is usually equivalent to `numpy.int32`.

Function argument names are expected to match the name of the argument in the low-level language (for languages such as Fortran, where such information can be used). If the function argument names are unknown please use positional-only arguments:

```python
def f(a: int, b: float, /): ...
```

### Name mapping with `@low_level`

You can use the `@low_level` decorator to explicitly map a Python name to its low-level implementation name. This is not strictly required, but it is recommended to avoid surprises as Pyccel can rename symbols internally (e.g. to avoid collisions). Such collisions are rare in stub files but using the decorator removes any possible ambiguity.

This decorator also allows you to rename functions and classes. For example functions can be mapped to names which are meaningful in Python such as magic method names (e.g. `__add__`).
Another common use case is for functions that accept different types of arguments. In Python stub files such functions are annotated with the [`@overload` decorator](https://typing.python.org/en/latest/spec/overload.html). Such functions may map to different low-level functions.

For example:

```python
from typing import overload
import numpy as np
from pyccel.decorators import low_level

@low_level('ffunc')
@overload
def func(a : np.float32): ...

@low_level('dfunc')
@overload
def func(a : np.float64): ...

@low_level('ifunc')
@overload
def func(a : np.int32): ...
```

### Compilation metadata

Pyccel requires some information about the underlying low-level code in order to be able to compile the generated wrapper. This information takes the form of metadata which is placed in comments, usually at the top of the file. The syntax for such metadata is:

```python
#$ header metavar key=val
```

Possible keys are:

- `includes` : Describes the include directories that must be passed to the compiler with the `-I` flag. This should be a string, folders are separated by commas.
- `libdirs` : Describes the library directories that must be passed to the compiler with the `-L` flag. This should be a string, folders are separated by commas.
- `libraries` : Describes the libraries which should be passed to the compiler with the `-l` flag. This should be a string, libraries are separated by commas.
- `flags` : Describes any additional compiler flags. This should be a string, options commas.
- `ignore_at_import` : Indicates that the library doesn't need to be imported (e.g. via a `use` statement in Fortran) in order to be used. This should be a boolean.

### Fortran specific rules

#### Functions

- Python functions with a single non-array result, match Fortran functions.

- Python functions with no results or multiple results, match Fortran subroutines.

- Argument names must match unless they are positional-only.

  Unless positional-only arguments are used, the Fortran printer prints the name of the argument being called. As a result, it is important that the argument names in the stub file match the argument names in the original code.

- Multiple returns are interpreted as multiple `intent(out)` arguments. These are always the first arguments of the Fortran function and are not called by name.

- Returning arrays is not currently recommended as the support is still quite restrictive.

  Array returns are interpreted as an `intent(out)` argument. If code calling such a method is translated, it is assumed that the array will be an allocatable and will be allocated with base-0 indexing. This output will be the first argument of the Fortran function. It will not be called by name.

- Lists, sets, and dictionaries are mapped to gFTL objects in the Fortran code.

- The name of the Python file must match the name of the module that should be imported to use this method.

  If there is no module to import (e.g. because the code is older than Fortran 90), this must be indicated with the appropriate metavariable:

  ```python
  #$ header metavar ignore_at_import=True
  ```

#### Classes

- The stub must define `__init__` and `__del__` methods, which map to the corresponding routines in the low-level code.

- The name stated in the `@low_level` decorator is the name of the type-bound procedure.

### C specific rules

- Multiple returns are interpreted as multiple `intent(out)` arguments. These are always the first arguments of the Fortran function and are not called by name.

- Arrays are mapped to instances of [STC](https://github.com/Stclib/STC)'s `cspan` class as described in the [documentation](ndarrays.md#the-n-dimensional-array-ndarray).

- Lists, sets, and dictionaries are mapped to STC objects in the C code.

- The name of the Python file must match the name of the header file that should be imported to use this method.

  If there is no header to include, this must be indicated with the appropriate metavariable:

  ```python
  #$ header metavar ignore_at_import=True
  ```

## Fortran Example

Suppose we have the following Fortran code that we want to be able to call from Python:

```fortran
module class_property
  use, intrinsic :: iso_c_binding, only : i64 => C_INT64_T, f64 => C_DOUBLE

  implicit none

  type :: Counter
    integer(i64) :: ncounters
    integer(i64), allocatable :: private_counters
  contains
    procedure :: create => counter_create
    procedure :: free => counter_free
    procedure :: increment_n => counter_increment_n
    procedure :: n_nonzero => counter_n_nonzero
    generic, public :: display => counter_display_element, counter_display_scaled
    procedure :: counter_display_element, counter_display_scaled
  end type Counter

contains

  subroutine counter_create(this, ncounters)
    class(Counter), intent(inout) :: this
    integer(i64), intent(in) :: ncounters
    this%ncounters = ncounters
    allocate(this%private_counters(ncounters))
    this%private_counters(:) = 0
  end subroutine counter_create

  subroutine counter_free(this)
    class(Counter), intent(inout) :: this
    deallocate(this%private_counters)
  end subroutine counter_free

  subroutine counter_increment_n(this, n)
    class(Counter), intent(inout) :: this
    integer(i64) :: n
    this%private_counters(n) = this%private_counters(n) + 1
  end subroutine counter_increment_n

  subroutine counter_display_element(this, n)
    class(Counter), intent(in) :: this
    integer(i64), value  :: n
    print *, "Counter value:", this%private_counters(n)
  end subroutine counter_display_element

  subroutine counter_display_scaled(this, scale)
    class(Counter), intent(in) :: this
    real(f64), value      :: scale
    integer :: i
    do i = 1, n
       print *, "Counter value (scaled):", real(this%private_counters(i), f64) * scale
    end do
  end subroutine counter_display_scaled


  function counter_n_nonzero(this) result(v)
    class(Counter), intent(in) :: this
    integer(i64) :: v
    integer :: i
    v = 0
    do i = 1, n
      v = v + this%private_counters(i)
    end do
  end function counter_n_nonzero

end module class_property
```

supposing the file is compiled to a library `libclass_property.so`, we can describe this code with the following stub file:

```python
#$ header metavar libraries="class_property"
#$ header metavar libdirs="."
from typing import overload
from pyccel.decorators import low_level

class Counter:
    ncounters : np.int64

    @low_level('create')
    def __init__(self, ncounters: int) -> None: ...

    @low_level('free')
    def __del__(self) -> None: ...

    @low_level('increment_n')
    def __iadd__(self, n : int) -> None: ...

    @property
    def n_nonzero(self) -> int: ...

    @low_level("counter_display_element")
    @overload
    def display(self, n: int) -> None: ...

    @low_level("counter_display_scaled")
    @overload
    def display(self, scale: float) -> None: ...
```

We then run `pyccel-wrap`

```bash
pyccel-wrap class_property.pyi
```

This generates a file `class_property.cpython-313-x86_64-linux-gnu.so` which is directly usable from Python.

More examples can be found in the [tests](https://github.com/pyccel/pyccel/tree/devel/tests/pyccel/wrap_scripts/fortran_tests). The stub files in this folder assume that the `.mod` files were saved into the sub-folder `__pyccel__mod__`.

## C Example

Suppose we have the following C code that we want to be able to call from Python:

```c
struct Counter {
    int64_t* private_counter_arr;
    int64_t value;
};

void Counter__create(struct Counter*, int64_t);
void Counter__free(struct Counter*);
void Counter__increment(struct Counter*, int64_t);
int64_t Counter__n_nonzero(struct Counter*);
void Counter__display_element(struct Counter*, int64_t);
void Counter__display_scaled(struct Counter*, double);
```

supposing the file is compiled to a library `libclass_property.so`, we can describe this code with the following stub file:

```python
#$ header metavar libraries="class_property"
#$ header metavar libdirs="."
from typing import overload
from pyccel.decorators import low_level

class Counter:
    @low_level('create')
    def __init__(self, start: int) -> None: ...

    @low_level('free')
    def __del__(self) -> None: ...

    @low_level('increment_n')
    def __iadd__(self, n : int) -> None: ...

    @low_level('counter_n_nonzero')
    @property
    def n_nonzero(self) -> int: ...

    @low_level("display_repeat")
    @overload
    def display(self, n: int) -> None: ...

    @low_level("display_scaled")
    @overload
    def display(self, scale: float) -> None: ...
```

We then run `pyccel-wrap`

```bash
pyccel-wrap class_property.pyi
```

This generates a file `class_property.cpython-313-x86_64-linux-gnu.so` which is directly usable from Python.

More examples can be found in the [tests](https://github.com/pyccel/pyccel/tree/devel/tests/pyccel/wrap_scripts/c_tests).
