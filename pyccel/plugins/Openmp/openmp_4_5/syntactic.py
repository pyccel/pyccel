from ..omp import OmpDirective, OmpClause, OmpEndDirective, OmpConstruct, OmpList, \
    OmpTxDirective, OmpTxEndDirective, OmpTxNode, OmpExpressionList
from ..omp import OmpScalarExpr, OmpIntegerExpr, OmpConstantPositiveInteger

def _treat_comment_line(self, line, expr, cls=None):
    """
    Parse a comment line.

    Parse a comment which fits in a single line if the comment
    begins with `#$omp` using textx.

    Parameters
    ----------
    self : object
        The parser self that is processing the code.
    line : str
        The comment line.
    expr : ast.Ast
        The comment object in the code. This is useful for raising
        errors.
    method : callable
        The fallback method to call if the line is not an OpenMP directive.
    cls : class, optional
        Used to access the configuration and class variables, defaults to None.

    Returns
    -------
    pyccel.ast.basic.PyccelAstNode
        The treated object as an Openmp node.

    See Also
    --------
    pyccel.plugins.Openmp.omp.OmpTxDirective : Class representing an OpenMP directive.
    pyccel.plugins.Openmp.omp.OmpTxEndDirective : Class representing an OpenMP end directive.
    """
    if line.startswith('#$') and line[2:].lstrip().startswith('omp'):
        from textx.exceptions import TextXError
        try:
            model = cls._omp_metamodel.model_from_str(line)
            directive = OmpTxEndDirective(model.statement, line, cls._version, lineno=expr.lineno,
                                          column=expr.col_offset) if model.statement.is_end_directive else OmpTxDirective(
                model.statement, line, cls._version, lineno=expr.lineno, column=expr.col_offset)
            return self._visit(directive)
        except TextXError as e:
            errors.report(e.message, severity="fatal", symbol=expr)
    else:
        return super(type(self), self)._treat_comment_line(line, expr)

def _visit(self, stmt, method, cls=None):
    """
    Visit a statement and determine if it should be skipped.

    This method processes AST statements and handles OpenMP directives.
    It manages the skipping of statements that are part of OpenMP constructs
    and ensures proper AST node creation.

    Parameters
    ----------
    self : object
        The parser self that is processing the code.
    stmt : ast.AST
        The statement to visit.
    method : callable
        The method to call for visiting the statement if it's not skipped.
    cls : class, optional
        The class to use for processing, defaults to None.

    Returns
    -------
    pyccel.ast.basic.PyccelAstNode
        The processed node, or an EmptyNode if the statement should be skipped.

    See Also
    --------
    pyccel.ast.core.EmptyNode : Node used for skipped statements.
    """
    if self._skip_stmts_count:
        self._skip_stmts_count -= 1
        return EmptyNode()
    else:
        res = method(stmt)
        if isinstance(stmt, OmpTxNode):
            res.set_current_ast(stmt.python_ast)
        return res

def _visit_OmpTxDirective(self, stmt, cls=None, method=None):
    if hasattr(self, f"_visit_{stmt.name.replace(' ', '_')}_directive"):
        return getattr(self, f"_visit_{stmt.name.replace(' ', '_')}_directive")(stmt)
    clauses = tuple(self._visit(clause) for clause in stmt.clauses)
    directive = OmpDirective(clauses=clauses, **stmt.get_fixed_state())
    if stmt.is_construct:
        body = []
        end = None
        container = None
        for el in self._context[::-1]:
            if isinstance(el, list):
                container = el[el.index(self._context[-2]) + 1:].copy()
                break
        for line in container:
            expr = self._visit(line)
            if isinstance(expr, OmpEndDirective) and stmt.name == expr.name:
                end = expr
                break
            body.append(expr)
        if end is None:
            errors.report(
                f"missing `end {stmt.name}` directive",
                symbol=stmt,
                severity="fatal",
            )
        self._skip_stmts_count = len(body) + 1
        body = CodeBlock(body=body)
        return OmpConstruct(start=directive, end=end, body=body)
    return directive

def _visit_for_directive(self, stmt, cls=None, method=None):
    loop = None
    for el in self._context[::-1]:
        if isinstance(el, list):
            loop_pos = el.index(self._context[-2]) + 1
            if len(el) < loop_pos + 1 or not isinstance(el[loop_pos], ast.For):
                errors.report(
                    f"{stmt.name} directive should be followed by a for loop",
                    symbol=stmt,
                    severity="fatal",
                )
            loop = self._visit(el[loop_pos])
            break
    clauses = tuple(self._visit(clause) for clause in stmt.clauses)
    directive = OmpDirective(clauses=clauses, **stmt.get_fixed_state())
    self._skip_stmts_count = 1
    body = CodeBlock(body=[loop])
    return OmpConstruct(start=directive, end=None, body=body)

def _visit_simd_directive(self, expr, cls=None, method=None):
    return self._visit_for_directive(expr)

def _visit_parallel_for_directive(self, expr, cls=None, method=None):
    return self._visit_for_directive(expr)

def _visit_parallel_for_simd_directive(self, expr, cls=None, method=None):
    return self._visit_for_directive(expr)

def _visit_target_teams_distribute_parallel_for_directive(self, expr, cls=None, method=None):
    return self._visit_for_directive(expr)

def _visit_OmpTxClause(self, expr, cls=None, method=None):
    omp_exprs = tuple(self._visit(e) for e in expr.omp_exprs)
    return OmpClause(omp_exprs=omp_exprs, **expr.get_fixed_state())

def _visit_OmpTxEndDirective(self, expr, cls=None, method=None):
    clauses = tuple(self._visit(clause) for clause in expr.clauses)
    return OmpEndDirective(clauses=clauses, **expr.get_fixed_state())

def _visit_OmpTxScalarExpr(self, expr, cls=None, method=None):
    fst = cls._helper_parse_expr(expr)
    return OmpScalarExpr(value=self._visit(fst), **expr.get_fixed_state())

def _visit_OmpTxIntegerExpr(self, expr, cls=None, method=None):
    fst = cls._helper_parse_expr(expr)
    return OmpIntegerExpr(value=self._visit(fst), **expr.get_fixed_state())

def _visit_OmpTxConstantPositiveInteger(self, expr, cls=None, method=None):
    fst = cls._helper_parse_expr(expr)
    return OmpConstantPositiveInteger(value=self._visit(fst), **expr.get_fixed_state())

def _visit_OmpTxList(self, expr, cls=None, method=None):
    fst = cls._helper_parse_expr(expr)
    return OmpList(value=self._visit(fst), **expr.get_fixed_state())

def _visit_OmpTxExpressionList(self, expr, cls=None, method=None):
    fst = cls._helper_parse_expr(expr)
    return OmpExpressionList(value=self._visit(fst), **expr.get_fixed_state())
