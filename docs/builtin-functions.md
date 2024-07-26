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
| `isinstance` | No |
| `issubclass` | No |
| `iter` | No |
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
| `round` | No |
| *`set`* | Python-only |
| `setattr` | No  |
| `slice` | No |
| `sorted` | No |
| `staticmethod` | No |
| `str` | No |
| **`sum`** | **Yes** |
| `super` | No |
| **`tuple`** | **Yes** |
| **`type`** | **Yes** |
| `vars` | No |
| **`zip`** | as a loop iterable |
| \_\_`import`\_\_ | No

## Dictionary methods

:warning: The dictionary support provided by Pyccel only covers unordered dictionaries.

| Method | Supported |
|----------|-----------|
| `clear` | No |
| `copy` | No |
| `get` | No |
| `items` | No |
| `keys` | No |
| `pop` | Python-only |
| `popitem` | No |
| `reversed` | No |
| `setdefault` | No |
| `update` | No |
| `values` | No |

