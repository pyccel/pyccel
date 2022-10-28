# coding: utf-8
# ------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
# ------------------------------------------------------------------------------------------#
"""
OpenMP has several constructs and directives, and this file contains the OpenMP types that are supported.
We represent some types with the OmpAnnotatedComment type.
These types are detailed on our documentation:
https://github.com/pyccel/pyccel/blob/master/tutorial/openmp.md
"""

from .basic import Basic

__all__ = (
    "OmpAnnotatedComment",
    # General
    "OmpConstruct",
    "OmpClause",
    "Omp",
    "OmpStatement",
    # Constructs
    "OmpParallelConstruct",
    "OmpEndConstruct",
    # Clauses
    "OmpIfClause",
    "OmpNumThreadsClause",
    "OmpDefaultClause",
    "OmpPrivateClause",
    "OmpFirstPrivateClause",
    "OmpSharedClause",
    "OmpCopyinClause",
    "OmpReductionClause",
    "OmpProcBindClause",
)


class OmpAnnotatedComment(Basic):

    """Represents an OpenMP Annotated Comment in the code."""

    __slots__ = ("_version",)
    _attribute_nodes = ()
    _current_omp_version = None

    def __init__(self, **kwargs):
        if self._current_omp_version is None:
            raise NotImplementedError(
                "OpenMP version not set (use OmpAnnotatedComment.set_current_version)"
            )
        self._version = float(kwargs.pop("version", "0") or "0")
        if self._version > self._current_omp_version:
            raise NotImplementedError(
                f"Syntax not supported in OpenMP version {self._current_omp_version}"
            )
        super().__init__(*kwargs)

    @property
    def version(self):
        """Returns the version of OpenMP syntax used."""
        return self._version

    @classmethod
    def set_current_version(cls, version):
        """Sets the version of OpenMP syntax to support."""
        cls._current_omp_version = version


class OmpConstruct(OmpAnnotatedComment):

    """Represents an OpenMP Construct in the code."""

    __slots__ = ("_name", "_clauses", "_clause_count", "_parent")
    _attribute_nodes = ("_clauses",)

    def __init__(self, **kwargs):
        self._name = kwargs.pop("name")
        self._clauses = kwargs.pop("clauses", [])
        self._parent = kwargs.pop("parent", None)

        # Count the occurrences of each clause
        # This is used to check for duplicate clauses if not allowed
        self._clause_count = {}
        for clause in self.clauses:
            if clause.name not in self._clause_count:
                self._clause_count[clause.name] = 0
            self._clause_count[clause.name] += 1

        super().__init__(**kwargs)

    @property
    def name(self):
        """Returns the name of the OpenMP construct."""
        return self._name

    @property
    def clauses(self):
        """Returns the clauses of the OpenMP construct."""
        return self._clauses

    @property
    def clauses_count(self):
        """Returns a dictionary containing the number of occurrences of each clause."""
        return self._clause_count

    def __str__(self):
        return f'{self.name} {" ".join(str(clause) for clause in self.clauses)}'

    def __repr__(self):
        return f'{self.name} {" ".join(repr(clause) for clause in self.clauses)}'


class OmpClause(OmpAnnotatedComment):
    """Represents an OpenMP Clause in the code."""

    __slots__ = ("_name", "_parent")
    _attribute_nodes = ()

    def __init__(self, **kwargs):
        self._name = kwargs.pop("name")
        self._parent = kwargs.pop("parent", None)
        super().__init__(**kwargs)

    @property
    def name(self):
        """Returns the name of the clause."""
        return self._name

    def __repr__(self):
        return f"{self.name}"


class Omp(OmpAnnotatedComment):
    """Represents a holder for all OpenMP statements."""

    __slots__ = ("_statements",)
    _attribute_nodes = ("_statements",)

    def __init__(self, **kwargs):
        self._statements = kwargs.pop("statements", [])
        super().__init__(**kwargs)

    @property
    def statements(self):
        """Returns the statements of the OpenMP holder."""
        return self._statements

    def __str__(self):
        return "\n".join(str(stmt) for stmt in self.statements)

    def __repr__(self):
        return "\n".join(repr(stmt) for stmt in self.statements)


class OmpStatement(OmpAnnotatedComment):
    """Represents an OpenMP statement."""

    __slots__ = ("_statement", "_parent")
    _attribute_nodes = ("_statement",)

    def __init__(self, **kwargs):
        self._statement = kwargs.pop("statement")
        self._parent = kwargs.pop("parent", None)
        super().__init__()

    @property
    def statement(self):
        """Returns the statement of the OpenMP statement."""
        return self._statement

    def __str__(self):
        return f"#$ omp {str(self.statement)}"

    def __repr__(self):
        return f"#$ omp {repr(self.statement)}"


class OmpParallelConstruct(OmpConstruct):
    """Represents an OpenMP Parallel Construct."""

    __slots__ = ()
    _attribute_nodes = ()


class OmpEndConstruct(OmpAnnotatedComment):
    """Represents an OpenMP End Construct."""

    __slots__ = ()
    _attribute_nodes = ()


class OmpIfClause(OmpClause):
    """Represents an OpenMP If Clause."""

    __slots__ = ()
    _attribute_nodes = ()


class OmpNumThreadsClause(OmpClause):
    """Represents an OpenMP NumThreads Clause."""

    __slots__ = ()
    _attribute_nodes = ()


class OmpDefaultClause(OmpClause):
    """Represents an OpenMP Default Clause."""

    __slots__ = ()
    _attribute_nodes = ()


class OmpPrivateClause(OmpClause):
    """Represents an OpenMP Private Clause."""

    __slots__ = ()
    _attribute_nodes = ()


class OmpFirstPrivateClause(OmpClause):
    """Represents an OpenMP FirstPrivate Clause."""

    __slots__ = ()
    _attribute_nodes = ()


class OmpSharedClause(OmpClause):
    """Represents an OpenMP Shared Clause."""

    __slots__ = ()
    _attribute_nodes = ()


class OmpCopyinClause(OmpClause):
    """Represents an OpenMP Copyin Clause."""

    __slots__ = ()
    _attribute_nodes = ()


class OmpReductionClause(OmpClause):
    """Represents an OpenMP Reduction Clause."""

    __slots__ = ()
    _attribute_nodes = ()


class OmpProcBindClause(OmpClause):
    """Represents an OpenMP ProcBind Clause."""

    __slots__ = ()
    _attribute_nodes = ()
