
def _visit_OmpDirective(self, expr):
    clauses = tuple(self._visit(clause) for clause in expr.clauses)
    directive = OmpDirective(clauses=clauses, **expr.get_fixed_state())
    return directive

def _visit_OmpConstruct(self, expr):
    if hasattr(self, f"_visit_{expr.start.name.replace(' ', '_')}_construct"):
        return getattr(self, f"_visit_{expr.start.name.replace(' ', '_')}_construct")(expr)

    body = self._visit(expr.body)
    start = self._visit(expr.start)
    end = self._visit(expr.end) if expr.end else None
    return OmpConstruct(start=start, end=end, body=body)

def _visit_for_construct(self, expr):
    body = self._visit(expr.body)
    start = self._visit(expr.start)
    return OmpConstruct(start=start, end=None, body=body)

def _visit_simd_construct(self, expr):
    return self._visit_for_construct(expr)

def _visit_parallel_for_simd_construct(self, expr):
    return self._visit_for_construct(expr)

def _visit_parallel_for_construct(self, expr):
    return self._visit_for_construct(expr)

def _visit_target_teams_distribute_parallel_for_construct(self, expr):
    return self._visit_for_construct(expr)

def _visit_OmpEndDirective(self, expr):
    if not isinstance(expr.current_user_node, OmpConstruct) and expr.is_construct:
        errors.report(
            f"End directive `{expr.name}` doesn't belong to any openmp construct",
            symbol=expr,
            severity="error",
        )
    clauses = tuple(self._visit(clause) for clause in expr.clauses)
    return OmpEndDirective(clauses=clauses, **expr.get_fixed_state())

def _visit_OmpScalarExpr(self, expr):
    value = self._visit(expr.value)
    if (
            not hasattr(value, "dtype")
            or (isinstance(value, FunctionCall) and not value.funcdef.results)
    ):
        errors.report(
            "expression needs to be a scalar expression",
            symbol=self,
            severity="fatal",
        )
    return OmpScalarExpr(value=value, **expr.get_fixed_state())

def _visit_OmpIntegerExpr(self, expr):
    value = self._visit(expr.value)
    if not hasattr(value, "dtype") or not isinstance(value.dtype, PythonNativeInt):
        errors.report(
            "expression must be an integer expression",
            symbol=self,
            severity="fatal",
        )
    return OmpIntegerExpr(value=value, **expr.get_fixed_state())

def _visit_OmpConstantPositiveInteger(self, expr):
    value = self._visit(expr.value)
    return OmpConstantPositiveInteger(value=value, **expr.get_fixed_state())

def _visit_OmpList(self, expr):
    items = tuple(self._visit(var) for var in expr.value)
    for i in items:
        if not isinstance(i, Variable):
            errors.report(
                "omp list must be a list of variables",
                symbol=expr,
                severity="fatal",
            )
    return OmpList(value=items, **expr.get_fixed_state())

def _visit_OmpExpressionList(self, expr):
    items = tuple(self._visit(var) for var in expr.value)
    for i in items:
        if not isinstance(i, Variable) and not isinstance(i, PyccelMinus) and not isinstance(i, PyccelAdd):
            errors.report(
                "omp list must be a list of expressions",
                symbol=expr,
                severity="fatal",
            )
    return OmpExpressionList(value=items, **expr.get_fixed_state())

def _visit_OmpClause(self, expr):
    omp_exprs = tuple(self._visit(e) for e in expr.omp_exprs)
    return OmpClause(omp_exprs=omp_exprs, **expr.get_fixed_state())
