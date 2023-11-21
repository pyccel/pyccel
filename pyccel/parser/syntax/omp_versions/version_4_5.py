# pylint: disable=protected-access, missing-function-docstring
"""
Module containing the grammar rules for the OpenMP 4.5 specification,
"""
from inspect import isclass
from abc import ABC, abstractmethod
import ast as Ast
from pyccel.ast.core import FunctionCall, Return
from pyccel.ast.variable import Variable
from pyccel.parser.extend_tree import extend_tree
from pyccel.ast.datatypes import NativeInteger, NativeVoid
from pyccel.ast.core import CodeBlock, For
from pyccel.ast.basic import Basic
from pyccel.ast.core import EmptyNode
from pyccel.errors.errors import Errors

__all__ = ("Openmp",)


errors = Errors()

class Openmp():
    """Class that groups all the OpenMP classes for constructs and clauses."""

    @classmethod
    def inner_classes_list(cls):
        """
        Return a list of all the classes that will be used to parse the OpenMP code.
        """
        results = []
        for attrname in dir(cls):
            obj = getattr(cls, attrname)
            if isclass(obj) and getattr(obj, '_used_in_grammar', False):
                results.append(obj)
        return results

    class OmpAnnotatedComment(Basic):

        """Represents an OpenMP Annotated Comment in the code."""

        __slots__ = ("_parent", "_raw", "_position", "_line", "VERSION", "DEPRECATED")
        _attribute_nodes = ()
        _current_omp_version = None

        def __init__(self, **kwargs):
            super().__init__()
            self._raw = kwargs.pop("raw", None)
            self._position = kwargs.pop('position', None)
            if self._position is None and hasattr(self, '_tx_position') and hasattr(self, '_tx_position_end'):
                self._position = (self._tx_position, self._tx_position_end)
            if not self._raw and not self._position:
                tmp = 0
            assert self._raw or self._position
            self._parent = kwargs.pop("parent", None)
            if self._current_omp_version is None:
                raise NotImplementedError(
                    "OpenMP version not set (use OmpAnnotatedComment.set_current_version)"
                )
            #self.VERSION = float(kwargs.pop("VERSION", "0") or "0")
            self.DEPRECATED = float(kwargs.pop("DEPRECATED", "inf") or "inf")
            self.VERSION = float(kwargs.pop("VERSION", 0.0) or 0.0)
            if isinstance(self.VERSION, list):
                self.VERSION = max(self.VERSION)

            if self.version > self._current_omp_version:
                raise NotImplementedError(
                    f"Syntax not supported in OpenMP version {self._current_omp_version}"
                )
            if self.deprecated <= self._current_omp_version:
                raise NotImplementedError(
                    f"Syntax deprecated in OpenMP version {self.DEPRECATED}"
                )

        @property
        def parent(self):
            """
            returns the parent of the omp object
            """
            omp_user_nodes = self.get_user_nodes((Openmp.OmpClause, Openmp.OmpDirective), excluded_nodes=(Openmp.OmpEndDirective))
            return omp_user_nodes[0] if len(omp_user_nodes) else self._parent

        @property
        def position(self):
            """
            retruns the position_start and position_end of an omp object's syntax inside the pragma.
            """
            return self._position

        @property
        def line(self):
            if self.fst:
                return self.fst.lineno
            elif isinstance(self.parent, Basic):
                return self.parent.line
            else:
                return None

        @property
        def raw(self):
            """
            Finds root model of the omp object and returns the object's syntax as written in the code.
            """
            if self._raw:
                return self._raw
            elif self._position:
                p = self
                while hasattr(p, 'parent'):
                    p = p.parent
                return p.raw[self.position[0]:self.position[1]]

        @property
        def version(self):
            """Returns the version of OpenMP syntax used."""
            return self.VERSION

        @property
        def deprecated(self):
            """Returns the deprecated version of OpenMP syntax used."""
            return self.DEPRECATED

        @classmethod
        def set_current_version(cls, version):
            """Sets the version of OpenMP syntax to support."""
            cls._current_omp_version = version

    class OmpConstruct(Basic):

        __slots__ = ("_start", "_end", "_body")
        _attribute_nodes = ("_start", "_end", "_body")

        def __init__(self, start, body, end=None):
            self._start = start
            self._end = end
            self._body = body
            super().__init__()

        @property
        def start(self):
            return self._start

        @property
        def body(self):
            return self._body

        @property
        def end(self):
            return self._end

    class OmpDirective(OmpAnnotatedComment):

        """
        Represents an every OpenMP Directive.

        Parameters
        ----------
        name : list
                A list of directives names

        name: str
              The name of the directive

        clauses: OmpClause
                  Clauses passed to the directive
        """
        __slots__ = ("_name", "_clauses", "_require_end_directive", "_tx_clauses", "_invalid_clauses")
        _used_in_grammar = True
        _attribute_nodes = ("_clauses",)

        def __init__(self, **kwargs):
            self._raw = None
            self._name = kwargs.pop("name")
            self._require_end_directive = kwargs.pop("require_end_directive", False)
            self._tx_clauses = kwargs.pop("_tx_clauses", [])
            self._tx_clauses = [c for c in self._tx_clauses if c]
            self._clauses = kwargs.pop("clauses", self._tx_clauses)
            self._invalid_clauses = kwargs.pop("_invalid_clauses", [])
            if any(not isinstance(c, Openmp.OmpClause) for c in self._clauses):
                self._clauses = [Openmp.OmpClause(position=(c._tx_position, c._tx_position_end),
                                                  omp_exprs=c.omp_exprs if hasattr(c, 'omp_exprs') else [],
                                                  name=c.name,
                                                  parent=c.parent,
                                                  allowed_parents=getattr(c, 'allowed_parents', None),
                                                  VERSION=getattr(c, 'VERSION', 0.0),
                                                  DEPRECATED=getattr(c, 'DEPRECATED', float('inf')))
                                 for c in self._clauses]
            super().__init__(**kwargs)
            if len(self._invalid_clauses):
                errors.report(
                   f"invalid clause `{self._invalid_clauses[0].name}` for `{self._name}` directive",
                    symbol=self,
                    column=self._invalid_clauses[0]._tx_position,
                    severity="fatal",
                   )

        @property
        def name(self):
            return self._name

        @property
        def clauses(self):
            return self._clauses

        @property
        def require_end_directive(self):
            return self._require_end_directive

        @property
        def raw(self):
            """
            Finds root model of the omp object and returns the object's syntax as written in the code.
            """
            # A workaround since textx gives wrong position for directives. It should be inherited instead.
            if self._raw:
                return self._raw
            else:
                p = self
                while hasattr(p, 'parent'):
                    p = p.parent
                return p.raw.replace("#$", "")

    class OmpEndDirective(OmpDirective):
        """Represents an OpenMP End Construct."""

        __slots__ = ("_coresponding_directive",)
        _attribute_nodes = ("_coresponding_directive",)
        _used_in_grammar = True

        def __init__(self, **kwargs):
            self._coresponding_directive = kwargs.pop("coresponding_directive", None)
            super().__init__(**kwargs)

        @property
        def coresponding_directive(self):
            return self._coresponding_directive


    class OmpClause(OmpAnnotatedComment):
        """Represents an OpenMP Clause in the code."""
        __slots__ = ("_omp_exprs", "_name")
        _attribute_nodes = ("_omp_exprs",)

        def __init__(self, **kwargs):
            self._omp_exprs = kwargs.pop('omp_exprs', [])
            if not isinstance(self._omp_exprs, tuple):
                self._omp_exprs = tuple(self._omp_exprs) if isinstance(self._omp_exprs, list) else (self._omp_exprs,)
            self._name = kwargs.pop('name', None)
            super().__init__(**kwargs)
            allowed_parents = kwargs.pop('allowed_parents', None)
            if allowed_parents:
                if isinstance(allowed_parents, str):
                    allowed_parents = (allowed_parents,)
                if self.parent.name not in allowed_parents:
                    errors.report(
                        f"invalid syntax `{self.name}` clause for `{self.parent.name}` directive",
                        symbol=self,
                        severity="fatal",
                    )


        @property
        def omp_exprs(self):
            """Returns the omp expressions of the clause."""
            return self._omp_exprs

        @property
        def name(self):
            """Returns the name of the clause"""
            return self._name

    class OmpExpr(OmpClause):

        __slots__ = ("_value",)
        _attribute_nodes = ()

        def __init__(self, **kwargs):
            self._value = kwargs.pop("value", None)
            super().__init__(**kwargs)

        @property
        def value(self):
            """Returns the name of the variable."""
            if self._value:
                return self._value
            else:
                return self.raw

    class OmpScalarExpr(OmpExpr):
        _used_in_grammar = True

    class OmpIntegerExpr(OmpExpr):
        _used_in_grammar = True

    class OmpConstantPositiveInteger(OmpExpr):
        _used_in_grammar = True

    class OmpList(OmpExpr):
        _used_in_grammar = True

        @property
        def raw(self):
            raw = super().raw
            return f"({raw},)"


    class SyntaxParser(ABC):

        def __init__(self):
            self._pending_directives = []
        @property
        def pending_directives(self):
            return self._pending_directives

        @abstractmethod
        def _visit(self, stmt):
            pass

        def _visit_OmpDirective(self, expr):
            clauses = [self._visit(clause) for clause in expr.clauses]
            directive = Openmp.OmpDirective(name=expr.name, clauses=clauses, raw=expr.raw, parent=expr.parent, require_end_directive=expr.require_end_directive)
            self._pending_directives.append(directive)
            return directive

        def _visit_OmpClause(self, expr):
            omp_exprs = tuple(self._visit(e) for e in expr.omp_exprs)
            return Openmp.OmpClause(name=expr.name, position=expr.position, omp_exprs=omp_exprs)

        def _visit_Omp(self, expr):
            statements = tuple(
                self._visit(stmt) for stmt in expr.statements
            )
            return statements

        def _find_coresponding_directive(self, end_directive):
            cor_directive = None
            if len(self._pending_directives) == 0:
                errors.report(
                   f"`end {end_directive.name}` misplaced",
                   symbol=end_directive,
                   severity="fatal",
                   )
            if end_directive.name != self._pending_directives[-1].name:
                if not self._pending_directives[-1].require_end_directive:
                    self._pending_directives.pop()
                    cor_directive = self._find_coresponding_directive(end_directive)
                else:
                    errors.report(
                        f"`end {end_directive.name}` misplaced",
                        symbol=end_directive,
                        severity="fatal",
                    )
            else:
                cor_directive = self._pending_directives.pop()

            return cor_directive

        def _visit_OmpEndDirective(self, expr):
            cor_directive = self._find_coresponding_directive(expr)
            clauses = [self._visit(clause) for clause in expr._clauses]
            return Openmp.OmpEndDirective(name=expr.name, clauses=clauses, coresponding_directive=cor_directive, raw=expr.raw, parent=expr.parent)

        def _visit_OmpScalarExpr(self, expr):
            fst = extend_tree(expr.value)
            if (
                    not isinstance(fst, Ast.Module)
                    or len(fst.body) != 1
                    or not isinstance(fst.body[0], Ast.Expr)
            ):
                errors.report(
                    "Invalid expression",
                    symbol=expr,
                    line=expr.line,
                    column=expr.position[0],
                    severity="fatal",
                )
            fst = fst.body[0].value
            return Openmp.OmpScalarExpr(value=self._visit(fst), raw=expr.raw)

        def _visit_OmpIntegerExpr(self, expr):
            fst = extend_tree(expr.value)
            if (
                    not isinstance(fst, Ast.Module)
                    or len(fst.body) != 1
                    or not isinstance(fst.body[0], Ast.Expr)
            ):
                errors.report(
                    "Invalid expression",
                    symbol=expr,
                    line=expr.line,
                    column=expr.position[0],
                    severity="fatal",
                )
            fst = fst.body[0].value
            return Openmp.OmpIntegerExpr(value=self._visit(fst), raw=expr.raw)

        def _visit_OmpConstantPositiveInteger(self, expr):
            fst = extend_tree(expr.value)
            if (
                    not isinstance(fst, Ast.Module)
                    or len(fst.body) != 1
                    or not isinstance(fst.body[0], Ast.Expr)
            ):
                errors.report(
                    "Invalid expression",
                    symbol=expr,
                    line=expr.line,
                    column=expr.position[0],
                    severity="fatal",
                )
            fst = fst.body[0].value
            return Openmp.OmpConstantPositiveInteger(value=self._visit(fst), raw=expr.raw)

        def _visit_OmpList(self, expr):
            fst = extend_tree(expr.value)
            if (
                    not isinstance(fst, Ast.Module)
                    or len(fst.body) != 1
                    or not isinstance(fst.body[0], Ast.Expr)
            ):
                errors.report(
                    "Invalid expression",
                    symbol=expr,
                    line=expr.line,
                    column=expr.position[0],
                    severity="fatal",
                )
            fst = fst.body[0].value
            return Openmp.OmpList(value=self._visit(fst), raw=expr.raw)

        def _visit_OmpExpr(self, expr):
            fst = extend_tree(expr.value)
            if (
                    not isinstance(fst, Ast.Module)
                    or len(fst.body) != 1
                    or not isinstance(fst.body[0], Ast.Expr)
            ):
                errors.report(
                    "Invalid expression",
                    symbol=expr,
                    line=expr.line,
                    column=expr.position[0],
                    severity="fatal",
                )
            fst = fst.body[0].value
            return Openmp.OmpExpr(value=self._visit(fst), raw=expr.raw)

    class SemanticParser(ABC):

        def __init__(self):
            self._omp_reserved_nodes = []

        @property
        def omp_reserved_nodes(self):
            return self._omp_reserved_nodes

        @abstractmethod
        def _visit(self, stmt):
            pass

        def _visit_Omp(self, expr):
            statements = tuple(
                self._visit(stmt) for stmt in expr.statements
            )
            return statements

        def _visit_OmpDirective(self, expr):
            if hasattr(self, f"_visit_{'_'.join(n for n in expr.name)}_directive"):
                return getattr(self, f"_visit_{'_'.join(n for n in expr.name)}_directive")(expr)
            clauses = [self._visit(clause) for clause in expr.clauses]
            directive = Openmp.OmpDirective(name=expr.name, clauses=clauses, raw=expr.raw, parent=expr.parent, require_end_directive=expr.require_end_directive)
            end_directives = [dirve for dirve in expr.get_all_user_nodes() if isinstance(dirve, Openmp.OmpEndDirective)]
            if len(end_directives) == 0:
               return directive
            assert len(end_directives) == 1
            end_dir = end_directives[0]
            end_dir.substitute(end_dir.coresponding_directive, EmptyNode())
            if end_dir.get_all_user_nodes() != expr.get_all_user_nodes():
                errors.report(
                    f"`end {end_dir.name}` directive misplaced",
                    symbol=end_dir,
                    severity="fatal",
                )
            container = expr.get_all_user_nodes()[-1]
            reserved_nodes = container.body[container.body.index(expr) + 1:container.body.index(end_dir) + 1]
            end_dir = self._visit(reserved_nodes[-1])
            body = self._visit(CodeBlock(reserved_nodes[:-1]))
            self._omp_reserved_nodes = reserved_nodes
            return Openmp.OmpConstruct(start=directive, end=end_dir, body=body)

        def _visit_for_directive(self, expr):
            clauses = [self._visit(clause) for clause in expr._clauses]
            directive = Openmp.OmpDirective(name=expr.name, clauses=clauses, raw=expr.raw, parent=expr.parent, require_end_directive=expr.require_end_directive)
            end_directives = [dirve for dirve in expr.get_all_user_nodes() if isinstance(dirve, Openmp.OmpEndDirective)]
            container = expr.get_user_nodes(CodeBlock)[0]
            if not isinstance(container.body[container.body.index(expr) + 1], For):
                errors.report(
                    f"{expr.name} directive should be followed by a for loop",
                    symbol=expr,
                    severity="fatal",
                )
            if len(end_directives) == 1:
                end_dir = end_directives[0]
                end_dir.substitute(end_dir.coresponding_directive, EmptyNode())
                if end_dir.get_all_user_nodes() != expr.get_all_user_nodes():
                    errors.report(
                        f"`end {end_dir.name}` directive misplaced",
                        symbol=end_dir,
                        severity="fatal",
                    )
                if not isinstance(container.body[container.body.index(expr) + 2], Openmp.OmpEndDirective):
                    errors.report(
                        f"`end {end_dir.name}` directive misplaced",
                        symbol=end_dir,
                        severity="fatal",
                    )
                reserved_nodes = container.body[container.body.index(expr) + 1:container.body.index(end_dir) + 1]
                end_dir = self._visit(reserved_nodes[-1])
                body = self._visit(CodeBlock(reserved_nodes[:-1]))
                self._omp_reserved_nodes = reserved_nodes
            else:
                end_dir = None
                reserved_nodes = container.body[container.body.index(expr) + 1:container.body.index(expr) + 2]
                body = self._visit(CodeBlock(reserved_nodes))
                self._omp_reserved_nodes = reserved_nodes

            return Openmp.OmpConstruct(start=directive, end=end_dir, body=body)

        def _visit_simd_directive(self, expr):
            return self._visit_for_directive(expr)

        def _visit_parallel_for_directive(self, expr):
            return self._visit_for_directive(expr)

        def _visit_OmpEndDirective(self, expr):
            if expr.coresponding_directive is None:
                errors.report(
                    f"{expr.name} does not match any directive",
                    symbol=expr,
                    severity="fatal",
                )
            clauses = [self._visit(clause) for clause in expr.clauses]
            return Openmp.OmpEndDirective(name=expr.name, clauses=clauses, raw=expr.raw, parent=expr.parent)

        def _visit_OmpScalarExpr(self, expr):
            value = self._visit(expr.value)
            if (
                not hasattr(value, "dtype")
                or isinstance(value.dtype, NativeVoid)
                or (isinstance(value, FunctionCall) and not value.funcdef.results)
            ):
                errors.report(
                    "expression needs to be a scalar expression",
                    symbol=self,
                    line=expr.line,
                    column=expr.position[0],
                    severity="fatal",
                )
            return Openmp.OmpScalarExpr(value=value, raw=expr.raw)

        def _visit_OmpIntegerExpr(self, expr):
            value = self._visit(expr.value)
            if not hasattr(value, "dtype") or not isinstance(value.dtype, NativeInteger):
                errors.report(
                    "expression must be an integer expression",
                    symbol=self,
                    line=expr.line,
                    column=expr.position[0],
                    severity="fatal",
                )
            return Openmp.OmpIntegerExpr(value=value, raw=expr.raw)

        def _visit_OmpConstantPositiveInteger(self, expr):
            value = self._visit(expr.value)
            return Openmp.OmpConstantPositiveInteger(value=value, raw=expr.raw)

        def _visit_OmpList(self, expr):
            items = tuple(self._visit(var) for var in expr.value)
            for i in items:
                if not isinstance(i, Variable):
                    errors.report(
                        "omp list must be a list of variables",
                        symbol=expr,
                        line=expr.line,
                        column=expr.position[0],
                        severity="fatal",
                    )
            return Openmp.OmpList(value=items, raw=expr.raw)

        def _visit_OmpClause(self, expr):
            #omp_exprs = tuple(self._visit(e) for e in expr.get_attribute_nodes(Openmp.OmpExpr))
            omp_exprs = tuple(self._visit(e) for e in expr.omp_exprs)
            return type(expr)(name=expr.name, omp_exprs=omp_exprs, raw=expr.raw)

    class CCodePrinter:

        def _print_OmpConstruct(self, expr):
            body = self._print(expr.body)
            if expr.end and expr.start.require_end_directive:
                return f"{self._print(expr.start)}\n{{\n{body}\n}}\n{self._print(expr.end)}\n"
            else:
                return f"{self._print(expr.start)}\n{body}\n"

        def _print_OmpDirective(self, expr):
            return f"#pragma {expr.raw}\n"

        def _print_OmpEndDirective(self, expr):
            return ""
