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

The stub file (`.pyi`) describes the low-level library in a way that is understandable in a Python environment. It follows the same conventions as standard Python type stub files (see the Python documentation), with a few additional rules specific to Pyccel:

- At the top of the file, you must include some metadata about the compilation

- If you are exposing a class to Python, the stub must define `__init__` and `__del__` methods, which map to the corresponding routines in the low-level code.

You must ensure that all functions, methods, and classes have precise type annotations that match the low-level signatures.
Particular care should be taken with integers. On most platforms Python's default integer precision is equivalent to `numpy.int64`, while the default integer precision in low-level languages like C and Fortran is usually equivalent to `numpy.int32`.
In addition to the general rules above there are more specific rules for [Fortran](#fortran-specific-rules) and [C](#c-specific-rules).

Function argument names are expected to match the name of the argument in the low-level language (for languages such as Fortran, where such information can be used). If the function argument names are unknown please use positional-only arguments:

```python
def f(a: int, b: float, /): ...
```

### Name mapping with `@low_level`

You can use the `@low_level` decorator to explicitly map a Python name to its low-level implementation name. This is not strictly required, but it is strongly recommended to avoid surprises as Pyccel can rename symbols internally (e.g. to avoid collisions). It also allows you to rename functions and classes.

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

- Argument names must match unless they are positional-only.

  Unless positional-only arguments are used the Fortran printer prints the name of the argument being called. As a result it is important that the argument names in the stub file match the argument names in the original code.

- Multiple returns are interpreted as multiple `intent(out)` arguments. These are always the first arguments of the Fortran function and are not called by name.

- Returning arrays is not recommended.

  Array returns are interpreted as an `intent(out)` argument. If code calling such a method is translated, it is assumed that the array will allocate with base-0 indexing. This output will be the first argument of the Fortran function. It will not be called by name.

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

- Arrays are mapped to instances of STC's `cspan` class.

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
    integer(i64) :: value
  contains
    procedure :: create => counter_create
    procedure :: free => counter_free
    procedure :: increment_n => counter_increment_n
    procedure :: get_value => counter_get_value
    generic, public :: display => counter_display_repeat, counter_display_scaled
    procedure :: counter_display_repeat, counter_display_scaled
  end type Counter

contains

  subroutine counter_create(this, start)
    class(Counter), intent(inout) :: this
    integer(i64), intent(in) :: start
    this%value = start
  end subroutine counter_create

  subroutine counter_free(this)
    class(Counter), intent(inout) :: this
    this%value = -1
  end subroutine counter_free

  subroutine counter_increment_n(this, n)
    class(Counter), intent(inout) :: this
    integer(i64) :: n
    this%value = this%value + n
  end subroutine counter_increment_n

  subroutine counter_display_repeat(this, n)
    class(Counter), intent(in) :: this
    integer(i64), value  :: n
    integer :: i
    do i = 1, n
       print *, "Counter value:", this%value
    end do
  end subroutine counter_display_repeat

  subroutine counter_display_scaled(this, scale)
    class(Counter), intent(in) :: this
    real(f64), value      :: scale
    print *, "Counter value (scaled):", real(this%value, f64) * scale
  end subroutine counter_display_scaled


  function counter_get_value(this) result(v)
    class(Counter), intent(in) :: this
    integer(i64) :: v
    v = this%value
  end function counter_get_value

end module class_property
```

supposing the file is compiled to a library `libclass_property.so`, we can describe this code with the following stub file:

```python
#$ header metavar libraries="class_property"
#$ header metavar libdirs="."
from typing import overload
from pyccel.decorators import low_level

class Counter:
    value : np.int64

    @low_level('create')
    def __init__(self, start: int) -> None: ...

    @low_level('free')
    def __del__(self) -> None: ...

    @low_level('increment_n')
    def __iadd__(self, n : int) -> None: ...

    @low_level('get_value')
    @property
    def my_value(self) -> int: ...

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

More examples can be found in the [tests](https://github.com/pyccel/pyccel/tree/devel/tests/pyccel/wrap_scripts/fortran_tests).

## C Example

Suppose we have the following C code that we want to be able to call from Python:

```c
struct Counter {
    int64_t value;
};

void Counter__create(struct Counter*, int64_t);
void Counter__free(struct Counter*);
void Counter__increment(struct Counter*, int64_t);
int64_t Counter__get_value(struct Counter*);
void Counter__display_repeat(struct Counter*, int64_t);
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

    @low_level('get_value')
    @property
    def value(self) -> int: ...

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

More examples can be found in the [tests](https://github.com/pyccel/pyccel/tree/devel/tests/pyccel/wrap_scripts/fortran_tests).
