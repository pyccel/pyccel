# Built-in functions

Python contains a limited number of builtin functions defined [here](https://docs.python.org/3/library/functions.html). Pyccel currently handles a small subset of those functions

| Function | Supported |
|----------|-----------|
| **`abs`** | **Yes** |
| `all` | No |
| `any` | No |
| `ascii` | No |
| `bin` | No |
| **`bool`** | **Yes** |
| `breakpoint` | No |
| `bytearray` | No |
| `bytes` | No |
| `callable` | No |
| `chr` | No |
| `classmethod` | No |
| `compile` | No |
| **`complex`** | **Yes** |
| `delattr` | No |
| *`dict`* | Preliminary Python and C **unordered** support |
| `dir` | No |
| `divmod` | No |
| **`enumerate`** | as a loop iterable |
| `eval` | No |
| `exec` | No |
| `filter` | No |
| **`float`** | **Yes** |
| `format` | No |
| `frozenset` | No |
| `getattr` | No |
| `globals` | No |
| `hasattr` | No |
| `hash` | No |
| `help` | No |
| `hex` | No |
| `id` | No |
| `input` | No |
| **`int`** | **Yes** |
| **`isinstance`** | **Yes** |
| `issubclass` | No |
| `iter` | No |
| **`lambda`** | Yes (but not as function arguments) |
| **`len`** | **Yes** |
| *`list`* | Python-only |
| `locals` | No |
| **`map`** | as a loop iterable |
| **`max`** | Full Fortran support and C support for 2 arguments |
| `memoryview` | No |
| **`min`** | Full Fortran support and C support for 2 arguments |
| `next` | No |
| `object` | No |
| `oct` | No |
| `open` | No |
| `ord` | No |
| `pow` | No |
| **`print`** | **Yes** |
| `property` | No |
| **`range`** | **Yes** |
| `repr` | No |
| `reversed` | No |
| **`round`** | **Yes** |
| **`set`** | **Yes** |
| `setattr` | No  |
| `slice` | No |
| `sorted` | No |
| `staticmethod` | No |
| `str` | As a type annotation or with string arguments |
| **`sum`** | **Yes** |
| `super` | No |
| **`tuple`** | **Yes** |
| **`type`** | **Yes** |
| `vars` | No |
| **`zip`** | as a loop iterable |
| `__import__` | No |

## List methods

| Method | Supported |
|----------|-----------|
| **`append`** | **Yes** |
| **`clear`** | **Yes** |
| `copy` | Python-only |
| `count` | No |
| **`extend`** | **Yes** |
| `index` | No |
| **`insert`** | **Yes** |
| `max` | No |
| `min` | No |
| **`pop`** | **Yes** |
| `remove` | Python-only |
| **`reverse`** | **Yes** |
| `sort` | Python-only |

## Set methods

| Method | Supported |
|----------|-----------|
| **`add`** | **Yes** |
| **`clear`** | **Yes** |
| **`copy`** | **Yes** |
| **`difference`** | **Yes** |
| **`difference_update`** | **Yes** |
| **`discard`** | **Yes** |
| **`intersection`** | **Yes** |
| **`intersection_update`** | **Yes** |
| **`isdisjoint`** | **Yes** |
| `issubset` | No |
| `issuperset` | No |
| **`pop`** | **Yes** |
| **`remove`** | **Yes** |
| `symmetric_difference` | No |
| `symmetric_difference_update` | No |
| **`union`** | **Yes** |
| **`update`** | **Yes** |

## Dictionary methods

:warning: The dictionary support provided by Pyccel only covers unordered dictionaries.

| Method | Supported |
|----------|-----------|
| **`clear`** | **Yes** |
| `copy` | Python-only |
| `get` | Python and C |
| **`items`** | **Yes** |
| **`keys`** | **Yes** |
| **`pop`** | **Yes** |
| **`popitem`** | **Yes** |
| `reversed` | No |
| `setdefault` | Python-only |
| `update` | No |
| **`values`** | **Yes** |

## String methods

| Method | Supported |
|----------|-----------|
| `capitalize` | No |
| `casefold` | No |
| `center` | No |
| `count` | No |
| `encode` | No |
| `endswith` | No |
| `expandtabs` | No |
| `find` | No |
| `format` | No |
| `format_map` | No |
| `index` | No |
| `isalnum` | No |
| `isalpha` | No |
| `isascii` | No |
| `isdecimal` | No |
| `isdigit` | No |
| `isidentifier` | No |
| `islower` | No |
| `isnumeric` | No |
| `isprintable` | No |
| `isspace` | No |
| `istitle` | No |
| `isupper` | No |
| `join` | No |
| `ljust` | No |
| `lower` | No |
| `lstrip` | No |
| `make_trans` | No |
| `partition` | No |
| `removeprefix` | No |
| `removesufix` | No |
| `replace` | No |
| `rfind` | No |
| `rindex` | No |
| `rjust` | No |
| `rpartition` | No |
| `rsplit` | No |
| `rstrip` | No |
| `split` | No |
| `splitlines` | No |
| `startswith` | No |
| `strip` | No |
| `swapcase` | No |
| `title` | No |
| `translate` | No |
| `upper` | No |
| `zfill` | No |
