# pylint: disable=protected-access
"""
C code generation for OpenMP 4.5 constructs.

Provides the mixin methods (added to the C printer via
`get_updated_codegen_methods`) that print OpenMP 4.5 AST nodes as `#pragma
omp` directives.
"""

from pyccel.plugins.Openmp.ast.omp import OmpConstruct

__all__ = (
    "_print_OmpConstruct",
    "_print_OmpDirective",
    "_print_OmpEndDirective",
)


def _print_OmpConstruct(self, expr):
    body = self._print(expr.body)
    if expr.end:
        return f"{self._print(expr.start)}{{\n{body}}}\n{self._print(expr.end)}"
    else:
        return f"{self._print(expr.start)}{body}"


def _print_OmpDirective(self, expr):
    return f"#pragma omp {expr.raw}\n"


def _print_OmpEndDirective(self, expr):
    if isinstance(expr.current_user_node, OmpConstruct):
        return ""
    else:
        return f"#pragma omp end {expr.raw}\n"
