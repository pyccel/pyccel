"""
Semantic-stage handling of OpenMP 5.0 directives.

Currently re-exports the `openmp_4_5.semantic` mixin methods unchanged, as
OpenMP 5.0 semantic handling has not yet diverged from 4.5.
"""

from ..openmp_4_5.semantic import (
    _visit_for_construct,
    _visit_OmpClause,
    _visit_OmpConstantPositiveInteger,
    _visit_OmpConstruct,
    _visit_OmpDirective,
    _visit_OmpEndDirective,
    _visit_OmpExpressionList,
    _visit_OmpIntegerExpr,
    _visit_OmpList,
    _visit_OmpScalarExpr,
    _visit_parallel_for_construct,
    _visit_parallel_for_simd_construct,
    _visit_simd_construct,
    _visit_target_teams_distribute_parallel_for_construct,
)

__all__ = (
    "_visit_OmpDirective",
    "_visit_OmpConstruct",
    "_visit_for_construct",
    "_visit_simd_construct",
    "_visit_parallel_for_simd_construct",
    "_visit_parallel_for_construct",
    "_visit_target_teams_distribute_parallel_for_construct",
    "_visit_OmpEndDirective",
    "_visit_OmpScalarExpr",
    "_visit_OmpIntegerExpr",
    "_visit_OmpConstantPositiveInteger",
    "_visit_OmpList",
    "_visit_OmpExpressionList",
    "_visit_OmpClause",
)
