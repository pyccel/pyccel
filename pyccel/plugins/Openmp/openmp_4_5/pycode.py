

__all__ = ("_print_OmpConstruct",
           "_print_OmpDirective",
           "_print_OmpEndDirective",)

def _print_OmpConstruct(self, expr):
    body = self._print(expr.body)
    start = self._print(expr.start)
    if expr.end:
        end = self._print(expr.end)
        return f"{start}\n{body}\n{end}\n"
    else:
        return f"{start}\n{body}\n"

def _print_OmpDirective(self, expr):
    return f"#$ omp {expr.raw}\n"

def _print_OmpEndDirective(self, expr):
    return f"#$ omp end {expr.raw}\n"
