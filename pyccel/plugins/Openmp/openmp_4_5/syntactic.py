# pylint: disable=protected-access
"""
Syntactic-stage parsing of OpenMP 4.5 directives.

Provides the mixin methods (added to the syntactic parser via
`get_updated_syntactic_methods`) that use the textx-based OpenMP 4.5 grammar
to parse OpenMP directives found in comments into `OmpTxNode` AST nodes.
"""

import ast
from os.path import dirname, join

from textx import metamodel_for_language
from textx.metamodel import metamodel_from_file

from pyccel.ast.core import CodeBlock, EmptyNode
from pyccel.errors.errors import Errors
from pyccel.parser.extend_tree import extend_tree
from pyccel.plugins.Openmp.ast.omp import (
    OmpClause,
    OmpConstantPositiveInteger,
    OmpConstruct,
    OmpDirective,
    OmpEndDirective,
    OmpExpressionList,
    OmpIntegerExpr,
    OmpList,
    OmpScalarExpr,
    OmpTxDirective,
    OmpTxEndDirective,
    OmpTxNode,
)

errors = Errors()

__all__ = (
    "__init__",
    "_helper_parse_expr",
    "_treat_comment_line",
    "_visit",
    "_visit_OmpTxDirective",
    "_visit_for_directive",
    "_visit_simd_directive",
    "_visit_parallel_for_directive",
    "_visit_parallel_for_simd_directive",
    "_visit_target_teams_distribute_parallel_for_directive",
    "_visit_OmpTxClause",
    "_visit_OmpTxEndDirective",
    "_visit_OmpTxScalarExpr",
    "_visit_OmpTxIntegerExpr",
    "_visit_OmpTxConstantPositiveInteger",
    "_visit_OmpTxList",
    "_visit_OmpTxExpressionList",
)


def __init__(self, *args, **kwargs):
    self._version = 4.5
    self._skip_stmts_count = 0
    cls = type(self)
    if not hasattr(cls, "_omp_metamodel"):
        this_folder = dirname(__file__)
        # Get metamodel from language description
        grammar = join(this_folder, "../grammar/openmp.tx")
        cls._omp_metamodel = metamodel_from_file(grammar)
        # object processors: are registered for particular classes (grammar rules)
        # and are called when the objects of the given class is instantiated.
        # The rules OMP_X_Y are used to insert the version of the syntax used
        textx_mm = metamodel_for_language("textx")
        grammar_model = textx_mm.grammar_model_from_file(grammar)

        def make_parent_processor(rule):
            """returns a processor that handles allowed parent directives"""
            return lambda _: rule.name.replace("_PARENT", "").lower()

        obj_processors = {
            r.name: make_parent_processor(r)
            for r in grammar_model.rules
            if r.name.endswith("_PARENT")
        }
        obj_processors.update(
            {
                "OMP_4_5": lambda _: 4.5,
                "OMP_5_0": lambda _: 5.0,
                "OMP_5_1": lambda _: 5.1,
                "TRUE": lambda _: True,
                "OMP_VERSION": lambda _: self._version,
            }
        )
        cls._omp_metamodel.register_obj_processors(obj_processors)
    super(type(self), self).__init__(*args, **kwargs)


def _helper_parse_expr(self, expr):
    """
    Parse an expression and returns the equivalent node.

    This method takes an OpenMP expression and converts it into a Python AST node
    using the extend_tree function. It performs validation to ensure the expression
    is valid and properly structured.

    Parameters
    ----------
    expr : str
        A python expression.

    Returns
    -------
    ast.AST
        The Python AST node equivalent to the input expression.
    """
    fst = extend_tree(expr.value)
    if (
        not isinstance(fst, ast.Module)
        or len(fst.body) != 1
        or not isinstance(fst.body[0], ast.Expr)
    ):
        errors.report(
            "Invalid expression",
            symbol=expr,
            severity="fatal",
        )
    return fst.body[0].value


def _treat_comment_line(self, line, expr):
    """
    Parse a comment line.

    Parse a comment which fits in a single line if the comment
    begins with `#$omp` using textx.

    Parameters
    ----------
    line : str
        The comment line.
    expr : ast.Ast
        The comment object in the code. This is useful for raising
        errors.

    Returns
    -------
    pyccel.ast.basic.PyccelAstNode
        The treated object as an Openmp node.

    See Also
    --------
    pyccel.plugins.Openmp.omp.OmpTxDirective : Class representing an OpenMP directive.
    pyccel.plugins.Openmp.omp.OmpTxEndDirective : Class representing an OpenMP end directive.
    """
    txt = line[1:].lstrip()
    if txt.startswith("$") and txt[1:].lstrip().startswith("omp"):
        from textx.exceptions import TextXError

        try:
            model = self._omp_metamodel.model_from_str(line)
            directive = (
                OmpTxEndDirective(
                    model.statement,
                    line,
                    self._version,
                    lineno=expr.lineno,
                    column=expr.col_offset,
                )
                if model.statement.is_end_directive
                else OmpTxDirective(
                    model.statement,
                    line,
                    self._version,
                    lineno=expr.lineno,
                    column=expr.col_offset,
                )
            )
            return self._visit(directive)
        except TextXError as e:
            raise errors.report(e.message, severity="fatal", symbol=expr)
    else:
        return super(type(self), self)._treat_comment_line(line, expr)


def _visit(self, stmt):
    """
    Visit a statement and determine if it should be skipped.

    This method processes AST statements and handles OpenMP directives.
    It manages the skipping of statements that are part of OpenMP constructs
    and ensures proper AST node creation.

    Parameters
    ----------
    stmt : ast.AST
        The statement to visit.

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
        res = super(type(self), self)._visit(stmt)
        if isinstance(stmt, OmpTxNode):
            res.set_current_ast(stmt.python_ast)
        return res


def _visit_OmpTxDirective(self, stmt):
    if hasattr(self, f"_visit_{stmt.name.replace(' ', '_')}_directive"):
        return getattr(self, f"_visit_{stmt.name.replace(' ', '_')}_directive")(stmt)
    clauses = tuple(self._visit(clause) for clause in stmt.clauses)
    directive = OmpDirective(clauses=clauses, **stmt.get_fixed_state())
    if stmt.is_construct:
        body = []
        end = None
        container = None
        for el in self._context[::-1]:
            if hasattr(el, "body"):
                container = el.body[el.body.index(self._context[-2]) + 1 :].copy()
                break
        for line in container:
            # Visit lines belonging to container
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
        # self._skip_stmts_count ensures that statements aren't visited twice
        self._skip_stmts_count = len(body) + 1
        body = CodeBlock(body=body)
        return OmpConstruct(start=directive, end=end, body=body)
    return directive


def _visit_for_directive(self, stmt):
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


def _visit_simd_directive(self, expr):
    return self._visit_for_directive(expr)


def _visit_parallel_for_directive(self, expr):
    return self._visit_for_directive(expr)


def _visit_parallel_for_simd_directive(self, expr):
    return self._visit_for_directive(expr)


def _visit_target_teams_distribute_parallel_for_directive(self, expr):
    return self._visit_for_directive(expr)


def _visit_OmpTxClause(self, expr):
    omp_exprs = tuple(self._visit(e) for e in expr.omp_exprs)
    return OmpClause(omp_exprs=omp_exprs, **expr.get_fixed_state())


def _visit_OmpTxEndDirective(self, expr):
    clauses = tuple(self._visit(clause) for clause in expr.clauses)
    return OmpEndDirective(clauses=clauses, **expr.get_fixed_state())


def _visit_OmpTxScalarExpr(self, expr):
    fst = self._helper_parse_expr(expr)
    return OmpScalarExpr(value=self._visit(fst), **expr.get_fixed_state())


def _visit_OmpTxIntegerExpr(self, expr):
    fst = self._helper_parse_expr(expr)
    return OmpIntegerExpr(value=self._visit(fst), **expr.get_fixed_state())


def _visit_OmpTxConstantPositiveInteger(self, expr):
    fst = self._helper_parse_expr(expr)
    return OmpConstantPositiveInteger(value=self._visit(fst), **expr.get_fixed_state())


def _visit_OmpTxList(self, expr):
    fst = self._helper_parse_expr(expr)
    return OmpList(value=self._visit(fst), **expr.get_fixed_state())


def _visit_OmpTxExpressionList(self, expr):
    fst = self._helper_parse_expr(expr)
    return OmpExpressionList(value=self._visit(fst), **expr.get_fixed_state())
