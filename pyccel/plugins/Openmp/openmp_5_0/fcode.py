"""
Fortran code generation for OpenMP 5.0 constructs.

Currently re-exports the `openmp_4_5.fcode` mixin methods unchanged, as
OpenMP 5.0 Fortran code generation has not yet diverged from 4.5.
"""

from ..openmp_4_5.fcode import (
    _helper_delay_clauses_printing,
    _print_end_section_directive,
    _print_for_construct,
    _print_OmpConstruct,
    _print_OmpDirective,
    _print_OmpEndDirective,
    _print_parallel_for_construct,
    _print_parallel_for_simd_construct,
    _print_simd_construct,
    _print_single_construct,
    _print_target_teams_distribute_parallel_for_construct,
)

__all__ = (
    "_helper_delay_clauses_printing",
    "_print_OmpConstruct",
    "_print_for_construct",
    "_print_single_construct",
    "_print_simd_construct",
    "_print_parallel_for_construct",
    "_print_parallel_for_simd_construct",
    "_print_target_teams_distribute_parallel_for_construct",
    "_print_OmpDirective",
    "_print_OmpEndDirective",
    "_print_end_section_directive",
)
