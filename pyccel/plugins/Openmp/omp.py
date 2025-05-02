import re
from pyccel.ast.basic import PyccelAstNode
from pyccel.errors.errors import Errors

errors = Errors()


class OmpAnnotatedComment(PyccelAstNode):
    """Parent class for all openmp classes, including classes used in the grammar."""

    __slots__ = ("_parent", "_raw", "_position", "_line", "_version", "_deprecated", "_omp_version")
    _attribute_nodes = ()

    def __init__(self, **kwargs):
        super().__init__()
        self._parent = kwargs.pop("parent", None)
        self._omp_version = kwargs.pop("omp_version", None)

        if self._omp_version is None:
            # fetch the omp version from the root model, it will not be fetched in after the syntactic stage
            p = self
            while hasattr(p, 'parent'):
                p = p.parent
            if hasattr(p, 'omp_version'):
                self._omp_version = p.omp_version
            else:
                raise NotImplementedError(
                    "OpenMP version not set"
                )
        self._raw = kwargs.pop("raw", None)
        self._position = kwargs.pop('position', None)
        self._line = kwargs.pop('line', None)
        if self._position is None and hasattr(self, '_tx_position') and hasattr(self, '_tx_position_end'):
            self._position = (self._tx_position, self._tx_position_end)
        assert self._raw or self._position
        self._version = kwargs.pop("VERSION", 0.0) or 0.0
        if isinstance(self._version, list):
            self._version = max(self._version)
        self._deprecated = float(kwargs.pop("DEPRECATED", "inf") or "inf")
        if self._version > self._omp_version:
            raise NotImplementedError(
                f"Syntax not supported in OpenMP version {self._omp_version}"
            )
        if self._deprecated <= self._omp_version:
            raise NotImplementedError(
                f"Syntax deprecated in OpenMP version {self._deprecated}"
            )

    @property
    def parent(self):
        """
        returns the parent of the omp object.
        """
        omp_user_nodes = self.get_user_nodes((OmpClause, OmpDirective, OmpConstruct),
                                             excluded_nodes=(OmpEndDirective))
        return omp_user_nodes[0] if len(omp_user_nodes) else self._parent

    @property
    def position(self):
        """
        returns the position_start and position_end of an omp object's syntax inside the pragma.
        """
        return self._position

    @property
    def line(self):
        """
        returns the line of an omp object from the root parent OmpDirective.
        """
        if self._line:
            return self._line
        p = self
        while not isinstance(p, OmpDirective):
            p = p.parent
        return p.python_ast.lineno

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
    def VERSION(self):
        """Returns the version of the OpenMP object's syntax used."""
        return self._version

    @property
    def deprecated(self):
        """Returns the deprecated version of OpenMP syntax used."""
        return self._deprecated

    @property
    def omp_version(self):
        """returns the openmp version used"""
        return self._omp_version

    def get_fixed_state(self):
        """Returns the attributes of an openmp object that do not change throughout the life of the object"""
        return {
            'position': self.position,
            'line': self.line,
            'raw': self.raw,
            'VERSION': self.VERSION,
            'deprecated': self.deprecated,
            'omp_version': self.omp_version,
        }

class OmpConstruct(PyccelAstNode):
    __slots__ = ("_start", "_end", "_body", "_omp_version")
    _attribute_nodes = ("_start", "_end", "_body")

    def __init__(self, start, body, end=None):
        self._start = start
        self._end = end
        self._body = body
        self._omp_version = start.omp_version
        super().__init__()

    @property
    def start(self):
        """Returns the directive that marks the start of the construct"""
        return self._start

    @property
    def body(self):
        """Returns a codeblock body of the construct"""
        return self._body

    @property
    def end(self):
        """Returns the end directive that marks the end of the construct"""
        return self._end

    @end.setter
    def end(self, end):
        """Used by the printers to tweak the expression of the construct"""
        self._end = end

    @property
    def name(self):
        return self._start.name

    @property
    def omp_version(self):
        """returns the openmp version used"""
        return self._omp_version

class OmpDirective(OmpAnnotatedComment):
    """
    Represents an OpenMP Directive.

    Parameters
    ----------
    name: str
          The name of the directive

    is_construct bool
          True if the directive is syntactically incorrect without a corresponding end directive

    clauses: OmpClause
              Clauses passed to the directive
    """
    __slots__ = ("_name", "_clauses", "_is_construct", "_tx_clauses", "_invalid_clauses")
    _attribute_nodes = ("_clauses",)

    def __init__(self, **kwargs):

        self._name = kwargs.pop("name")
        self._is_construct = kwargs.pop("is_construct", False)
        self._clauses = kwargs.pop("clauses", [])
        super().__init__(**kwargs)

    @property
    def name(self):
        """Returns The name of the Directive"""
        return self._name

    @property
    def clauses(self):
        """Returns The clauses of the Directive"""
        return self._clauses

    @property
    def is_construct(self):
        """Returns True if the Directive is a construct"""
        return self._is_construct

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
            return re.sub(r"#\s*\$\s*omp\s*(end)?", "", p.raw)

    def has_clause(self, clause_name):
        """checks if the directive has a clause with the name clause_name"""
        return any(c.name == clause_name for c in self.clauses)

    @classmethod
    def from_directive(cls, directive, **kwargs):
        """Takes a directive and returns a directive that keeps its unchangeable state"""
        clauses = kwargs.pop('clauses', directive.clauses)
        parent = kwargs.pop('parent', directive.parent)
        coresponding_directive = kwargs.pop('coresponding_directive', None)
        d_attrs = directive.get_fixed_state()
        return cls(clauses=clauses,
                   is_construct=directive.is_construct,
                   name=directive.name,
                   parent=parent,
                   coresponding_directive=coresponding_directive,
                   **d_attrs, )

    @classmethod
    def from_tx_directive(cls, directive, fst=None):
        """Takes a tx object that represent a directive and returns a directive"""
        is_construct = directive.is_construct
        if is_construct is None:
            is_construct = False

        # Imposed by the grammar: clean up the tx directive object's clauses.
        tx_clauses = getattr(directive, '_tx_clauses', [])
        tx_clauses = [c for c in tx_clauses if c]
        tx_clauses = [c.clause if hasattr(c, 'clause') else c for c in tx_clauses]
        # Transform the tx clauses to OmpClause
        clauses = [OmpClause.from_tx_clause(c, parent=directive) if not isinstance(c, OmpList) else c for
                   c in tx_clauses]
        # Get a list containing all the versions of the childs omp objcets,
        # necessary to calculate the directives versions.
        omp_objs = [directive] + tx_clauses
        VERSION = [getattr(o, 'VERSION') for o in omp_objs if getattr(o, 'VERSION', None)]
        VERSION = [v for l_v in VERSION for v in (l_v if isinstance(l_v, list) else [l_v])]
        # Get the position of the directive from tx attributes.
        position = (directive._tx_position, directive._tx_position_end)
        # Invalid clauses are syntactically correct omp clauses, captured within the current directive, but are invalid
        # clauses in the context of the current directive
        invalid_clauses = getattr(directive, "_invalid_clauses", [])
        if len(invalid_clauses):
            errors.report(
                f"invalid clause `{invalid_clauses[0].name}` for `{directive.name}` directive",
                symbol=directive,
                column=position[0],
                severity="fatal",
            )
        if directive.is_end_directive:
            return OmpEndDirective(clauses=clauses,
                                          name=directive.name,
                                          parent=directive.parent,
                                          VERSION=VERSION,
                                          DEPRECATED=getattr(directive, 'deprecated', float('inf')),
                                          position=position,
                                          fst=fst,
                                          line=None,
                                          raw=None)
        else:
            return cls(clauses=clauses,
                       is_construct=is_construct,
                       name=directive.name,
                       parent=directive.parent,
                       VERSION=VERSION,
                       DEPRECATED=getattr(directive, 'deprecated', float('inf')),
                       position=position,
                       fst=fst,
                       line=None,
                       raw=None)

class OmpEndDirective(OmpDirective):
    """Represents an OpenMP End Directive."""

    __slots__ = ("_coresponding_directive",)
    _attribute_nodes = ("_coresponding_directive",)


    def __init__(self, **kwargs):
        self._coresponding_directive = kwargs.pop("coresponding_directive", None)
        super().__init__(**kwargs)

    @property
    def coresponding_directive(self):
        return self._coresponding_directive


class OmpClause(OmpAnnotatedComment):
    """Represents an OpenMP Clause in the code."""
    __slots__ = ("_omp_exprs", "_name", "_allowed_parents")
    _attribute_nodes = ("_omp_exprs",)

    def __init__(self, **kwargs):
        self._omp_exprs = kwargs.pop('omp_exprs', [])
        if not isinstance(self._omp_exprs, list):
            self._omp_exprs = [self.omp_exprs]
        self._name = kwargs.pop('name', None)
        super().__init__(**kwargs)
        # check if the parent directive accepts the current clause.
        self._allowed_parents = kwargs.pop('allowed_parents', None)
        if self._allowed_parents is not None:
            if self.parent.name not in self._allowed_parents:
                errors.report(
                    f"invalid syntax `{self.name}` clause for `{self.parent.name}` directive",
                    symbol=self,
                    severity="fatal",
                )

    @classmethod
    def from_clause(cls, clause, **kwargs):
        """Takes a clause and returns a clause that keeps its unchangeable state"""
        omp_exprs = kwargs.pop('omp_exprs', clause.omp_exprs)
        parent = kwargs.pop('parent', clause.parent)
        d_attrs = clause.get_fixed_state()
        return cls(omp_exprs=omp_exprs,
                   name=clause.name,
                   parent=parent,
                   allowed_parents=clause.allowed_parents,
                   **d_attrs, )

    @classmethod
    def from_tx_clause(cls, tx_clause, **kwargs):
        """Takes a tx object that represent a clause and returns an OmpClause"""
        # Get the all the Allowed parents of the omp objects used inside the clause.
        # sets allowed parents to None if there are no restrictions about the allowed_parents
        parent = kwargs.pop('parent', tx_clause.parent)
        omp_objs = [tx_clause] + [getattr(tx_clause, attr_name) for attr_name in tx_clause._tx_attrs]
        allowed_parents = [set(getattr(obj, 'allowed_parents')) for obj in omp_objs
                           if getattr(obj, 'allowed_parents', None)]
        allowed_parents = set.intersection(*allowed_parents) if len(allowed_parents) > 0 else None
        # Get the position of the clause
        position = (tx_clause._tx_position, tx_clause._tx_position_end)
        return cls(position=position,
                   omp_exprs=getattr(tx_clause, 'omp_exprs', []),
                   name=getattr(tx_clause, 'name', None),
                   parent=parent,
                   allowed_parents=allowed_parents,
                   VERSION=getattr(tx_clause, 'VERSION', 0.0),
                   DEPRECATED=getattr(tx_clause, 'deprecated', float('inf')),
                   raw=None,
                   line=None)

    @property
    def omp_exprs(self):
        """Returns the omp expressions of the clause."""
        return self._omp_exprs

    @property
    def name(self):
        """Returns the name of the clause"""
        return self._name

    @property
    def allowed_parents(self):
        """Returns the allowed parent directives of the clause, None if there are no restrictions"""
        return self._allowed_parents


class OmpExpr(OmpClause):
    """Parent class of omp object that should represent a python expression"""

    __slots__ = ("_value",)
    _attribute_nodes = ()

    def __init__(self, **kwargs):
        self._value = kwargs.pop("value", None)
        super().__init__(**kwargs)

    @property
    def value(self):
        """Returns the expression, or the tweaked raw that should represent a python expression"""
        if self._value:
            return self._value
        else:
            return self.raw

    @classmethod
    def from_omp_expr(cls, expr, **kwargs):
        """From an OmpExpr returns an OmpExpr that keeps its unchangeable state"""
        value = kwargs.pop('value', expr.value)
        d_attrs = expr.get_fixed_state()
        return cls(value=value, **d_attrs)


class OmpScalarExpr(OmpExpr):
    """Represents a Scalar"""
    pass


class OmpConstantPositiveInteger(OmpExpr):
    """Represents a constant positive integer"""
    pass


class OmpIntegerExpr(OmpExpr):
    """Represents an integer"""
    pass


class OmpList(OmpExpr):
    """Represents a list"""

    @property
    def raw(self):
        """Returns a tweaked raw that represents a tuple"""
        raw = super().raw
        return f"({raw},)"
