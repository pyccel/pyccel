"""
Fortran code generation for OpenMP 4.5 constructs.

Provides the mixin methods (added to the Fortran printer via
`get_updated_codegen_methods`) that print OpenMP 4.5 AST nodes as `!$omp`
directives, including a helper for constructs whose clauses must be moved
from the start directive to the end directive.
"""

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


@classmethod
def _helper_delay_clauses_printing(cls, start, end, clauses):
    """
    Transfer clauses of a directive to an OmpEndDirective for printing.

    This method handles the special case in Fortran where certain clauses
    need to be moved from the start directive to the end directive for proper
    printing. It modifies the raw representation of the directives accordingly.

    Parameters
    ----------
    start : OmpDirective
        The starting directive of an OpenMP construct.
    end : OmpEndDirective or None
        The ending directive of an OpenMP construct, or None if there is no end directive.
    clauses : list of str
        Names of clauses to be moved from the start-to-end directive.

    Returns
    -------
    tuple
        A tuple containing (modified_start_string, modified_end_string).

    See Also
    --------
    FCodePrinter._print_for_construct : Method that uses this helper.
    FCodePrinter._print_single_construct : Method that uses this helper.

    Examples
    --------
    >>> start = OmpDirective(clauses=[OmpClause(name='nowait')])
    >>> end = None
    >>> start_str, end_str = FCodePrinter._helper_delay_clauses_printing(start, end, ['nowait'])
    >>> print(start_str)
    !$omp
    >>> print(end_str)
    !$omp end nowait
    """
    clauses = tuple(c for c in start.clauses if c.name in clauses)
    if clauses or end:
        if end:
            end = (
                f"!$omp end {end.name} {' '.join(c.raw for c in end.clauses + clauses)}"
            )
        else:
            end = f"!$omp end {start.name} {' '.join(c.raw for c in clauses)}"
    start = start.raw
    for c in clauses:
        start = start.replace(c.raw, "", 1)
    start = f"!$omp {start}\n"
    return start, end


def _print_OmpConstruct(self, expr):
    if hasattr(self, f"_print_{expr.start.name.replace(' ', '_')}_construct"):
        return getattr(self, f"_print_{expr.start.name.replace(' ', '_')}_construct")(
            expr
        )
    body = self._print(expr.body)
    start = self._print(expr.start)
    end = self._print(expr.end)
    return f"{start}\n{body}\n{end}\n"


def _print_for_construct(self, expr):
    start, end = self._helper_delay_clauses_printing(expr.start, expr.end, ["nowait"])
    start = re.sub(r"\bfor\b", "do", start)
    body = self._print(expr.body)
    if end:
        end = re.sub(r"\bfor\b", "do", end)
        return f"{start}\n{body}\n{end}\n"
    else:
        return f"{start}\n{body}\n"


def _print_single_construct(self, expr):
    start, end = self._helper_delay_clauses_printing(
        expr.start, expr.end, ["nowait", "copyprivate"]
    )
    body = self._print(expr.body)
    return f"{start}\n{body}\n{end}\n"


def _print_simd_construct(self, expr):
    return self._print_for_construct(expr)


def _print_parallel_for_construct(self, expr):
    return self._print_for_construct(expr)


def _print_parallel_for_simd_construct(self, expr):
    return self._print_for_construct(expr)


def _print_target_teams_distribute_parallel_for_construct(self, expr):
    return self._print_for_construct(expr)


def _print_OmpDirective(self, expr):
    return f"!$omp {expr.raw}\n"


def _print_OmpEndDirective(self, expr):
    if hasattr(self, f"_print_end_{expr.name.replace(' ', '_')}_directive"):
        return getattr(self, f"_print_end_{expr.name.replace(' ', '_')}_directive")(
            expr
        )
    return f"!$omp end {expr.raw}\n"


def _print_end_section_directive(self, expr):
    return ""
