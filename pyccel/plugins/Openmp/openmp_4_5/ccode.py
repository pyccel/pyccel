__all__ = (
    "_print_OmpConstruct",
    "_print_OmpDirective",
    "_print_OmpEndDirective",
)


def _print_OmpConstruct(self, expr):
    body = self._print(expr.body)
    if expr.end:
        return f"{self._print(expr.start)}\n{{\n{body}\n}}\n{self._print(expr.end)}\n"
    else:
        return f"{self._print(expr.start)}\n{body}\n"


def _print_OmpDirective(self, expr):
    return f"#pragma omp {expr.raw}\n"


def _print_OmpEndDirective(self, expr):
    if isinstance(expr.current_user_node, OmpConstruct):
        return ""
    else:
        return f"#pragma omp end {expr.raw}\n"
