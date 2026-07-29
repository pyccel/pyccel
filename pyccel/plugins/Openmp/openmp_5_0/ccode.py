"""
C code generation for OpenMP 5.0 constructs.

Currently re-exports the `openmp_4_5.ccode` mixin methods unchanged, as
OpenMP 5.0 C code generation has not yet diverged from 4.5.
"""

from ..openmp_4_5.ccode import (
    _print_OmpConstruct,
    _print_OmpDirective,
    _print_OmpEndDirective,
)

__all__ = (
    "_print_OmpConstruct",
    "_print_OmpDirective",
    "_print_OmpEndDirective",
)
