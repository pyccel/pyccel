"""
OpenMP 4.5 support for the OpenMP pyccel plugin.

Aggregates the submodules implementing OpenMP 4.5 parsing (`syntactic`,
`semantic`) and code generation (`ccode`, `fcode`, `pycode`).
"""

from . import ccode, fcode, pycode, semantic, syntactic
