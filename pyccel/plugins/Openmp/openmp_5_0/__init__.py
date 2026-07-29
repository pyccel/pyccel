"""
OpenMP 5.0 support for the OpenMP pyccel plugin.

Aggregates the submodules implementing OpenMP 5.0 parsing (`syntactic`,
`semantic`) and code generation (`ccode`, `fcode`, `pycode`). Each submodule
currently re-exports the corresponding `openmp_4_5` module unchanged.
"""

from . import ccode, fcode, pycode, semantic, syntactic
