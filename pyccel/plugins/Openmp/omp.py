"""Classes and methods that handle supported Openmp objects"""

import re
from ast import AST

from pyccel.ast.basic import PyccelAstNode
from pyccel.errors.errors import Errors

errors = Errors()


class OmpAst(AST):
    """New AST node representing an OPENMP syntax"""

    def __init__(self, lineno, col_offset):
        super().__init__()
        self.lineno = lineno
        self.col_offset = col_offset


class OmpNode(PyccelAstNode):
    """
    Parent class for all openmp classes, including classes used in the grammar.

    Parameters
    ----------
    raw : str
        The raw syntax of the section of the comment that represent the object, as provided in the source code.

    position : tuple
        The start and end positions of the OpenMP syntax in the source code, used to print errors.

    """

    _attribute_nodes = ()

    def __init__(self, raw, position, **kwargs):
        super().__init__()
        self._raw = raw
        self._position = position

    @property
    def position(self):
        """
        returns the position_start and position_end of an omp object's syntax inside the pragma.
        """
        return self._position

    @property
    def raw(self):
        """
        Finds root model of the omp object and returns the object's syntax as written in the code.
        """
        return self._raw

    def get_fixed_state(self):
        """Returns the attributes of an openmp object that do not change throughout the life of the object"""
        return {
            'position': self.position,
            # 'line': self.line,
            'raw': self.raw,
        }


class OmpConstruct(OmpNode):
    """
    Represents an OpenMP Construct.

    An OpenMP construct is a block of code , delimited by an OpenMP Directive at the
    start and potentially an end directive.

    Parameters
    ----------
    start : OmpDirective
        The directive that marks the start of the construct.

    body : CodeBlock
        The block of code (code statements) or syntax forming the body within the construct.

    end : OmpEndDirective, optional
        The directive that marks the end of the construct, if applicable.
    """
    _attribute_nodes = ("_start", "_end", "_body")

    def __init__(self, start, body, end=None):
        self._start = start
        self._end = end
        self._body = body
        super().__init__(raw=start.raw, position=start.position)

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

    @property
    def name(self):
        """Returns the name of the construct"""
        return self._start.name


class OmpDirective(OmpNode):
    """
    Represents an OpenMP Directive.

    Parameters
    ----------
    name: str
          The name of the directive

    is_construct bool
          True if the directive is syntactically incorrect without a corresponding end directive

    clauses: tuple of OmpClause
              Clauses passed to the directive
    """
    _attribute_nodes = ("_clauses",)

    def __init__(self, name, clauses, is_construct, raw,
                 position, **kwargs):
        self._name = name
        self._is_construct = is_construct
        self._clauses = clauses
        super().__init__(raw, position, **kwargs)

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

    def get_fixed_state(self):
        d_fixed = super().get_fixed_state()
        return {
            **d_fixed,
            'name': self.name,
            'is_construct': self.is_construct,
        }

    def has_clause(self, clause_name):
        """checks if the directive has a clause with the name clause_name"""
        return any(c.name == clause_name for c in self.clauses)


class OmpEndDirective(OmpDirective):
    """Represents an OpenMP End Directive."""


class OmpClause(OmpNode):
    """
    Represents an OpenMP Clause.

    Parameters
    ----------
    name: str
          The name of the clause

    omp_exprs: tuple
          OpenMP expressions passed to the clause
    """

    _attribute_nodes = ("_omp_exprs",)

    def __init__(self, name, omp_exprs, raw, position, **kwargs):
        self._omp_exprs = omp_exprs
        self._name = name
        super().__init__(raw, position, **kwargs)

    @property
    def omp_exprs(self):
        """Returns the omp expressions of the clause."""
        return self._omp_exprs

    @property
    def name(self):
        """Returns the name of the clause"""
        return self._name

    def get_fixed_state(self):
        """Returns the attributes of an openmp clause that do not change throughout the life of the object"""
        d_fixed = super().get_fixed_state()
        return {
            'name': self.name,
            **d_fixed,
        }


class OmpExpr(OmpNode):
    """
    Parent class of omp object that should represent a python expression

    Parameters
    ----------
    value: str or any
          The value or Python expression represented by this object
    """

    _attribute_nodes = ()

    def __init__(self, value, raw, position, **kwargs):
        self._value = value
        super().__init__(raw, position, **kwargs)

    @property
    def value(self):
        """Returns the expression, or the tweaked raw that should represent a python expression"""
        if self._value:
            return self._value
        else:
            return self.raw


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
    def value(self):
        """Returns the expression, or the tweaked raw that should represent a python expression"""
        if self._value:
            return self._value
        else:
            return f"({self.raw},)"


class OmpTxNode(OmpNode):
    """
    Parent class for all textx openmp classes, common grammar logic is treated here.

    Parameters
    ----------
    tx_obj: a textx object

    comment: str
        The python comment that represents the openmp syntax.

    omp_version : float
        The version of OpenMP chosen by the user.

    lineno : int
        The line number in the source code where the OpenMP syntax occurs.

    column : int
        The column number in the source code where the OpenMP syntax of the current object occurs.

    version : float | list
        The OpenMP version required for the annotated comment.
        If the syntax contains sections with different versions, they are stored in a list by textx.

    deprecated : float
        The OpenMP version in which the syntax was deprecated. Defaults to infinity.

    """

    _attribute_nodes = ()

    def __init__(self, tx_obj, comment, lineno=None, column=None, omp_version=None, version=None, deprecated=None,
                 **kwargs):
        self._omp_version = omp_version
        self._parent = tx_obj.parent

        if self._omp_version is None:
            # fetch the omp version from the root model, it will not be fetched in after the syntactic stage
            p = self._parent
            while hasattr(p, 'parent'):
                p = p.parent
            if hasattr(p, 'omp_version'):
                self._omp_version = p.omp_version
            else:
                raise NotImplementedError(
                    "OpenMP version not set"
                )
        position = (tx_obj._tx_position, tx_obj._tx_position_end)
        raw = comment[position[0]:position[1]]

        self._version = version or 0.0
        if isinstance(version, list):
            self._version = max(version)
        self._deprecated = deprecated or float("inf")
        super().__init__(raw=raw, position=position, **kwargs)
        if lineno is not None and column is not None:
            self.set_current_ast(OmpAst(lineno, self.position[0] + column))
        if self.version > self.omp_version:
            errors.report(
                f"Syntax not supported in OpenMp version {self.omp_version}",
                symbol=self,
                severity="warning",
            )
        if self.deprecated <= self.omp_version:
            errors.report(
                f"Syntax deprecated in OpenMp version {self.omp_version}",
                symbol=self,
                severity="warning",
            )

    @property
    def version(self):
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

    @property
    def parent(self):
        """
        returns the parent of the omp object.
        """
        return self._parent


class OmpTxDirective(OmpTxNode, OmpDirective):
    """
    Represents an OpenMP textx Directive. Common grammar logic for a directive is treated here.

    Parameters
    ----------
    tx_directive: a textx directive object

    """

    _attribute_nodes = ("_clauses",)

    def __init__(self, tx_directive, comment, lineno=None, column=None, **kwargs):
        d_attrs = {attr_name: getattr(tx_directive, attr_name) for attr_name in tx_directive._tx_attrs}
        is_construct = d_attrs.get('is_construct')
        if is_construct is None:
            is_construct = False
        # Imposed by the grammar: clean up the tx directive object's clauses.
        clauses = d_attrs.get('_tx_clauses', [])
        # Get a list containing all the versions of the childs omp objcets,
        # necessary to calculate the directives versions.
        version = d_attrs.get('VERSION')
        version = max(filter(None, [*[c.VERSION for c in clauses if hasattr(c, 'VERSION')], version]), default=None)
        name = d_attrs.get('name')
        clauses = [c for c in clauses if c]
        clauses = [c.clause if hasattr(c, 'clause') else c for c in clauses]
        clauses = [OmpTxClause(c, comment, lineno=lineno, column=column) for c in clauses]

        super().__init__(tx_obj=tx_directive, comment=comment, lineno=lineno, column=column, name=name, clauses=clauses,
                         is_construct=is_construct, omp_version=d_attrs.get('omp_version'), version=version,
                         deprecated=d_attrs.get('DEPRECATED'), **kwargs)
        self._raw = re.sub(r"#\s*\$\s*omp\s*(end)?", "", comment)

        # Invalid clauses are syntactically correct omp clauses, captured within the current directive, but are invalid
        # clauses in the context of the current directive
        invalid_clauses = d_attrs.get('_invalid_clauses', [])
        if len(invalid_clauses):
            errors.report(
                f"invalid clause `{invalid_clauses[0].name}` for `{name}` directive",
                symbol=self,
                column=self._position[0],
                severity="fatal",
            )


class OmpTxEndDirective(OmpTxDirective):
    """Represents an OpenMP End Directive."""
    pass


class OmpTxClause(OmpTxNode, OmpClause):
    """
    Represents an OpenMP textx Clause.

    Parameters
    ----------
    tx_clause: a textx clause object

    """

    _attribute_nodes = ("_omp_exprs",)

    def __init__(self, tx_clause, comment, lineno=None, column=None, **kwargs):
        d_attrs = {attr_name: getattr(tx_clause, attr_name) for attr_name in tx_clause._tx_attrs}
        omp_exprs = getattr(tx_clause, 'omp_exprs', tuple())
        if not isinstance(omp_exprs, tuple):
            omp_exprs = (omp_exprs,) if omp_exprs else tuple()
        omp_exprs = tuple(
            globals().get(ex.__class__.__name__)(ex, comment, lineno=lineno, column=column) for ex in omp_exprs)
        allowed_parents = d_attrs.get('allowed_parents', tuple())
        allowed_parents = [set(attr.allowed_parents) for attr in [*d_attrs.values(), *omp_exprs] if
                           hasattr(attr, 'allowed_parents')] + ([set(allowed_parents)] if len(allowed_parents) else [])
        allowed_parents = set.intersection(*allowed_parents) if len(allowed_parents) > 0 else None
        super().__init__(tx_obj=tx_clause, comment=comment, lineno=lineno, column=column, name=d_attrs.get('name'),
                         omp_exprs=omp_exprs, omp_version=d_attrs.get('omp_version'), version=d_attrs.get('VERSION'),
                         deprecated=d_attrs.get('DEPRECATED'), **kwargs)
        if hasattr(self.parent, 'clause'):
            self._parent = self.parent.parent

        # check if the parent directive accepts the current clause.
        if allowed_parents is not None:
            if self.parent.name not in allowed_parents:
                errors.report(
                    f"invalid syntax `{self.name}` clause for `{self.parent.name}` directive",
                    symbol=self,
                    severity="fatal",
                )


class OmpTxExpr(OmpTxNode, OmpExpr):
    """
    Parent class of textx OpenMp object that should represent a python expression

    Parameters
    ----------
    tx_obj: a textx obj

    """

    _attribute_nodes = ()

    def __init__(self, tx_obj, comment, lineno=None, column=None, **kwargs):
        d_attrs = {attr_name: getattr(tx_obj, attr_name) for attr_name in tx_obj._tx_attrs}
        super().__init__(tx_obj=tx_obj, comment=comment, lineno=lineno, column=column, value=d_attrs.get('value'),
                         omp_version=d_attrs.get('omp_version'), version=d_attrs.get('VERSION'),
                         deprecated=d_attrs.get('DEPRECATED'), **kwargs)


class OmpTxScalarExpr(OmpTxExpr, OmpScalarExpr):
    """Represents a Scalar"""

    def __init__(self, tx_expr, comment, **kwargs):
        super().__init__(tx_obj=tx_expr, comment=comment, **kwargs)


class OmpTxConstantPositiveInteger(OmpTxExpr, OmpConstantPositiveInteger):
    """Represents a constant positive integer"""

    def __init__(self, tx_expr, comment, **kwargs):
        super().__init__(tx_obj=tx_expr, comment=comment, **kwargs)


class OmpTxIntegerExpr(OmpTxExpr, OmpIntegerExpr):
    """Represents an integer"""

    def __init__(self, tx_expr, comment, **kwargs):
        super().__init__(tx_obj=tx_expr, comment=comment, **kwargs)


class OmpTxList(OmpTxExpr, OmpList):
    """Represents a list"""

    def __init__(self, tx_expr, comment, **kwargs):
        super().__init__(tx_obj=tx_expr, comment=comment, **kwargs)
