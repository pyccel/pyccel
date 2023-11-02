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

class OmpMeta(type):
    def __init__(cls, name, bases, clsdict):
        super().__init__(name, bases, clsdict)
        cls.clauses_cls_factory()

class Openmp(metaclass=OmpMeta):
    """Class that groups all the OpenMP classes for constructs and clauses."""

    all_clauses = [
            'OmpIfClause',
            'OmpNumThreadsClause',
            'OmpDefaultClause',
            'OmpPrivateClause',
            'OmpFirstPrivateClause',
            'OmpSharedClause',
            'OmpCopyinClause',
            'OmpReductionClause',
            'OmpProcBindClause',
            'OmpLastPrivateClause',
            'OmpLinearClause',
            'OmpScheduleClause',
            'OmpCollapseClause',
            'OmpOrderedClause',
            'OmpNoWaitClause',
            'OmpCopyPrivateClause',
            'OmpSafeLenClause',
            'OmpSimdLenClause',
            'OmpAlignedClause',
            'OmpFinalClause',
            'OmpUntiedClause',
            'OmpMergeableClause',
            'OmpDependClause',
            'OmpPriorityClause',
            'OmpGrainSizeClause',
            'OmpNumTasksClause',
            'OmpDeviceClause',
            'OmpMapClause',
            'OmpDefaultMapClause',
            'OmpIsDevicePtrClause',
            'OmpNumTeamsClause',
            'OmpThreadLimitClause',
            'OmpDistScheduleClause',
    ]
    all_directives = {
        'parallel': {
            'allowed_clauses': ("if", "num_threads", "default", "private", "firstprivate", "shared", "copyin",
                                "reduction", "proc_bind",),
            'deprecated_clauses': (),
            'require_end_directive': True,
        },
        'for': {
            'allowed_clauses': ("private", "firstprivate", "lastprivate", "linear", "reduction", "schedule", "collapse",
                                "ordered", "nowait",),
            'deprecated_clauses': (),
            'require_end_directive': False,
        },
        'sections': {
            'allowed_clauses': ("private", "firstprivate", "lastprivate", "linear", "reduction", "nowait",),
            'deprecated_clauses': (),
            'require_end_directive': True,
        },
        'section': {
            'allowed_clauses': (),
            'deprecated_clauses': (),
            'require_end_directive': False,
        },
        'single': {
            'allowed_clauses': ("private", "firstprivate", "copyprivate", "nowait",),
            'deprecated_clauses': (),
            'require_end_directive': True,
        },
        'simd': {
            'allowed_clauses': ("safelen", "simdlen", "linear", "aligned", "private", "lastprivate", "reduction",
                                "collapse"),
            'deprecated_clauses': (),
            'require_end_directive': False,
        },
        'task': {
            'allowed_clauses': ("if", "final", "untied", "default", "mergeable", "private", "shared", "depend",
                                "priority"),
            'deprecated_clauses': (),
            'require_end_directive': True,
        },
        'taskloop': {
            'allowed_clauses': ("if", "shared", "private", "firstprivate", "lastprivate", "default", "grainsize",
                                "num_tasks", "collapse", "final", "priority", "untied", "mergeable", "nogroup"),
            'deprecated_clauses': (),
            'require_end_directive': False,
        },
        'taskyield': {
            'allowed_clauses': (),
            'deprecated_clauses': (),
            'require_end_directive': False,
        },

        'target': {
            'allowed_clauses': ("if", "device", "private", "firstprivate", "map", "is_device_ptr", "defaultmap",
                                "nowait", "depend"),
            'deprecated_clauses': (),
            'require_end_directive': True,
        },

        'teams': {
            'allowed_clauses': ("num_teams", "thread_limit", "default", "private", "firstprivate", "shared",
                                "reduction"),
            'deprecated_clauses': (),
            'require_end_directive': True,
        },

        'distribute': {
            'allowed_clauses': ("private", "firstprivate", "lastprivate", "collapse", "dist_schedule"),
            'deprecated_clauses': (),
            'require_end_directive': False,
        },

        'master':{
            'allowed_clauses': (),
            'deprecated_clauses': (),
            'require_end_directive': True,
        },

        'critical': {
            'allowed_clauses': (),
            'deprecated_clauses': (),
            'require_end_directive': True,
        },

        'barrier': {
            'allowed_clauses': (),
            'deprecated_clauses': (),
            'require_end_directive': False,
        },

        'taskwait': {
            'allowed_clauses': (),
            'deprecated_clauses': (),
            'require_end_directive': False,
        },

        'taskgroup': {
            'allowed_clauses': (),
            'deprecated_clauses': (),
            'require_end_directive': True,
        },

        'flush': {
            'allowed_clauses': (),
            'deprecated_clauses': (),
            'require_end_directive': False,
        },
        'ordered': {
            'allowed_clauses': ("OmpThreadsClause", "OmpSimdClause"),
            'deprecated_clauses': (),
            'require_end_directive': True,
        },

        'cancel': {
            'allowed_clauses': ("OmpIfClause",),
            'deprecated_clauses': (),
            'require_end_directive': False,
        },
        'atomic': {
            'allowed_clauses': (),
            'deprecated_clauses': (),
            'require_end_directive': False,
        },
        'parallel for': {
            'disallowed_clauses':   ('OmpNoWaitClause',),
            'require_end_directive': False,
        },
        'parallel sections': {
            'disallowed_clauses': ('OmpNoWaitClause',),
        },
        'for simd': {
            'disallowed_clauses': (),
            'require_end_directive': False,
        },
        'parallel for simd': {
            'disallowed_clauses': ('OmpNoWaitClause',),
            'require_end_directive': False,
        },
    }
    @classmethod
    def clauses_cls_factory(cls):

        for clause_name in cls.all_clauses:
            if not hasattr(cls, clause_name):
                clause = type(clause_name, (cls.OmpClause,), {"_used_in_grammar": True})
                setattr(cls, clause_name, clause)

    @classmethod
    def inner_classes_list(cls):
        """
        Return a list of all the classes that will be used to parse the OpenMP code.
        """
        results = []
        for attrname in dir(cls):
            obj = getattr(cls, attrname)
            if isclass(obj) and getattr(obj, '_used_in_grammar', False) is True:
                results.append(obj)
        return results

    class OmpAnnotatedComment(Basic):

        """Represents an OpenMP Annotated Comment in the code."""

        __slots__ = ("VERSION", "DEPRECATED")
        _attribute_nodes = ()
        _current_omp_version = None

        def __init__(self, **kwargs):
            if self._current_omp_version is None:
                raise NotImplementedError(
                    "OpenMP version not set (use OmpAnnotatedComment.set_current_version)"
                )
            self.VERSION = float(kwargs.pop("VERSION", "0") or "0")
            self.DEPRECATED = float(kwargs.pop("DEPRECATED", "inf") or "inf")
            if self.version > self._current_omp_version:
                raise NotImplementedError(
                    f"Syntax not supported in OpenMP version {self._current_omp_version}"
                )
            if self.deprecated <= self._current_omp_version:
                raise NotImplementedError(
                    f"Syntax deprecated in OpenMP version {self.DEPRECATED}"
                )
            super().__init__()

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

        """Represents an OpenMP Construct in the code."""

        __slots__ = ("_name", "_dname", "_clauses", "_raw", "_allowed_clauses", "_deprecated_clauses", "_require_end_directive")
        _used_in_grammar = True

        def __init__(self, **kwargs):
            self._dname = tuple(kwargs.pop("dname"))
            self._name = ' '.join(n for n in self._dname)
            self._clauses = kwargs.pop("clauses", [])
            self._raw = kwargs.pop("raw", "")
            if len(self._dname) == 1:
                self._allowed_clauses = Openmp.all_directives[self._name]['allowed_clauses']
                self._deprecated_clauses = Openmp.all_directives[self._name]['deprecated_clauses']
                self._require_end_directive = Openmp.all_directives[self._name]['require_end_directive']
            else:
                self._allowed_clauses = (set.union(*(set(Openmp.all_directives[d]['allowed_clauses'])
                                                for d in self._dname)) - set(Openmp.all_directives[self._name]['disallowed_clauses']),)

                self._deprecated_clauses = set.union(*(set(Openmp.all_directives[d]['deprecated_clauses']) for d in self._dname))

                self._require_end_directive = Openmp.all_directives[self._name]['require_end_directive']
            super().__init__(**kwargs)


        @property
        def name(self):
            return self._name
        @property
        def dname(self):
            return self._dname

        @property
        def clauses(self):
            return self._clauses

        @property
        def allowed_clauses(self):
            return self._allowed_clauses

        @property
        def deprecated_clauses(self):
            return self._deprecated_clauses

        @property
        def require_end_directive(self):
            return self._require_end_directive

        @property
        def raw(self):
            return self._raw

        @raw.setter
        def raw(self, value):
            self._raw = value

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

    class OmpExpr(Basic):
        __slots__ = ("_value",)
        _attribute_nodes = ()

        def __init__(self, **kwargs):
            self._value = kwargs.pop("value")
            super().__init__()

        @property
        def value(self):
            """Returns the name of the variable."""
            return self._value

    class OmpScalarExpr(OmpExpr):
        _used_in_grammar = True

    class OmpIntegerExpr(OmpExpr):
        _used_in_grammar = True

    class OmpConstantPositiveInteger(OmpExpr):
        _used_in_grammar = True

    class OmpList(OmpExpr):
        _used_in_grammar = True

    class OmpClause(OmpAnnotatedComment):
        """Represents an OpenMP Clause in the code."""

        __slots__ = ("_name", "_omp_exprs")
        _attribute_nodes = ("_omp_exprs",)
        def __init__(self, **kwargs):
            self._name = kwargs.pop("name")
            self._omp_exprs = tuple(kwargs.pop(e) for e in list(kwargs.keys()) if isinstance(kwargs[e], Openmp.OmpExpr))
            super().__init__(**kwargs)

        @property
        def name(self):
            """Returns the name of the clause."""
            return self._name

        @property
        def omp_exprs(self):
            """Returns the omp expressions of the clause."""
            return self._omp_exprs

    class SyntaxParser(ABC):

        def __init__(self):
            self._pending_directives = []
            for clause in Openmp.all_clauses:
                method_name = f"_visit_{clause}"
                if not hasattr(self, method_name):
                    setattr(self, method_name, self._visit_clause_factory())
        @property
        def pending_directives(self):
            return self._pending_directives

        def _visit_clause_factory(self):
            def _visit(expr):
                for att in expr.get_attribute_nodes(Openmp.OmpExpr):
                    expr.substitute(att, self._visit(att))
                return expr
            return _visit

        @abstractmethod
        def _visit(self, stmt):
            pass

        def _visit_OmpDirective(self, expr):
            clauses = [self._visit(clause) for clause in expr.clauses]
            #expr.substitute(expr.clauses, clauses)
            directive = Openmp.OmpDirective(dname=expr.dname, clauses=clauses, raw=expr.raw)
            self._pending_directives.append(directive)
            return directive
        def _visit_Omp(self, expr):
            statements = tuple(
                self._visit(stmt) for stmt in expr.statements
            )
            return statements

        def _find_coresponding_directive(self, end_directive):
            cor_directive = None
            if len(self._pending_directives) == 0:
                errors.report(
                   f"`{end_directive}` misplaced",
                   symbol=end_directive,
                   severity="fatal",
                   )
            if end_directive.dname != self._pending_directives[-1].dname:
                if not self._pending_directives[-1].require_end_directive:
                    self._pending_directives.pop()
                    cor_directive = self._find_coresponding_directive(end_directive)
                else:
                    errors.report(
                        f"`{end_directive}` misplaced",
                        symbol=end_directive,
                        severity="fatal",
                    )
            else:
                cor_directive = self._pending_directives.pop()

            return cor_directive

        def _visit_OmpEndDirective(self, expr):
            cor_directive = self._find_coresponding_directive(expr)
            clauses = [self._visit(clause) for clause in expr._clauses]
            return Openmp.OmpEndDirective(dname=expr.dname, clauses=clauses, coresponding_directive=cor_directive, raw=expr.raw)

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
                    severity="fatal",
                )
            fst = fst.body[0].value
            return Openmp.OmpScalarExpr(value=self._visit(fst))

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
                    severity="fatal",
                )
            fst = fst.body[0].value
            return Openmp.OmpIntegerExpr(value=self._visit(fst))

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
                    severity="fatal",
                )
            fst = fst.body[0].value
            return Openmp.OmpConstantPositiveInteger(value=self._visit(fst))

        def _visit_OmpList(self, expr):
            fst = extend_tree(expr.value.__str__())
            if (
                    not isinstance(fst, Ast.Module)
                    or len(fst.body) != 1
                    or not isinstance(fst.body[0], Ast.Expr)
            ):
                errors.report(
                    "Invalid expression",
                    symbol=expr,
                    severity="fatal",
                )
            fst = fst.body[0].value
            return Openmp.OmpList(value=self._visit(fst))

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
                    severity="fatal",
                )
            fst = fst.body[0].value
            return Openmp.OmpExpr(value=self._visit(fst))

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
            if hasattr(self, f"_visit_{expr.dname}_directive"):
                return getattr(self, f"_visit_omp_{expr.dname}")(expr)
            clauses = [self._visit(clause) for clause in expr.clauses]
            directive = Openmp.OmpDirective(dname=expr.dname, clauses=clauses, raw=expr.raw)
            end_directives = [dirve for dirve in expr.get_all_user_nodes() if isinstance(dirve, Openmp.OmpEndDirective)]
            if len(end_directives) == 0:
               return directive
            assert len(end_directives) == 1
            end_dir = end_directives[0]
            end_dir.substitute(end_dir.coresponding_directive, EmptyNode())
            if end_dir.get_all_user_nodes() != expr.get_all_user_nodes():
                errors.report(
                    f"{expr} and {end_dir}, should be contained in the same block",
                    symbol=expr,
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
            directive = Openmp.OmpDirective(dname=expr.dname, clauses=clauses, raw=expr.raw)
            end_directives = [dirve for dirve in expr.get_all_user_nodes() if isinstance(dirve, Openmp.OmpEndDirective)]
            container = expr.get_user_nodes(CodeBlock)[0]
            if not isinstance(container.body[container.body.index(expr) + 1], For):
                errors.report(
                    f"{expr} should be followed by a for loop",
                    symbol=expr,
                    severity="fatal",
                )
            if len(end_directives) == 1:
                end_dir = end_directives[0]
                end_dir.substitute(end_dir.coresponding_directive, EmptyNode())
                if end_dir.get_all_user_nodes() != expr.get_all_user_nodes():
                    errors.report(
                        f"{expr} and {end_dir}, should be contained in the same block",
                        symbol=expr,
                        severity="fatal",
                    )
                if not isinstance(container.body[container.body.index(expr) + 2], Openmp.OmpEndDirective):
                    errors.report(
                        f"{end_dir} misplaced",
                        symbol=expr,
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

        def _visit_parallelfor_directive(self, expr):
            return self._visit_for_directive(expr)

        def _visit_OmpEndDirective(self, expr):
            if expr.coresponding_directive is None:
                errors.report(
                    f"OMP END does not match any OMP {expr.dname}",
                    symbol=expr,
                    severity="fatal",
                )
            clauses = [self._visit(clause) for clause in expr.clauses]
            return Openmp.OmpEndDirective(dname=expr.dname, clauses=clauses, raw=expr.raw)

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
                    severity="fatal",
                )
            return Openmp.OmpScalarExpr(value=value)

        def _visit_OmpIntegerExpr(self, expr):
            value = self._visit(expr.value)
            if not hasattr(value, "dtype") or not isinstance(value.dtype, NativeInteger):
                errors.report(
                    "expression must be an integer expression",
                    symbol=self,
                    severity="fatal",
                )
            return Openmp.OmpIntegerExpr(name=expr.name, omp_exprs=(value,))

        def _visit_OmpConstantPositiveInteger(self, expr):
            value = self._visit(expr.value)
            return Openmp.OmpConstantPositiveInteger(value=value)

        def _visit_OmpList(self, expr):
            items = tuple(self._visit(var) for var in expr.value)
            for i in items:
                if not isinstance(i, Variable):
                    errors.report(
                        "OMP PRIVATE clause must be a list of variables",
                        symbol=expr,
                        severity="fatal",
                    )
            return Openmp.OmpList(value=items)

        def _visit_OmpClause(self, expr):
            omp_exprs = (self._visit(e) for e in expr.get_attribute_nodes(Openmp.OmpExpr))
            return Openmp.OmpClause(name=expr.name, omp_exprs=omp_exprs)

    class CCodePrinter:

        def _print_OmpConstruct(self, expr):
            body = self._print(expr.body)
            if expr.end and expr.start.require_end_directive:
                return f"{self._print(expr.start)}\n{{\n{body}\n}}\n{self._print(expr.end)}\n"
            else:
                return f"{self._print(expr.start)}\n{body}\n"

        def _print_OmpDirective(self, expr):
            return f"{expr.raw}\n"

        def _print_OmpEndDirective(self, expr):
            return ""