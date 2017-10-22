# coding: utf-8

__all__ = ["BasicStmt"]

class BasicStmt(object):
    """
    Base class for all objects in Pyccel.
    """

    def __init__(self, **kwargs):
        """
        Constructor for the base class.

        Conventions:

        1) Every extension class must provide the properties stmt_vars and
        local_vars
        2) stmt_vars describes the list of all variables that are
        created by the statement.
        3) local_vars describes the list of all local variables to the
        statement, like the index of a For statement.
        4) Every extension must implement the update function. This function is
        called to prepare for the applied property (for example the expr
        property.)

        Parameters
        ==========
        statements : list
            list of statements from pyccel.types.ast
        unallocated: dict
            a dictionary containing all unallocated variables. later we will
            have to traverse the AST and allocate them. a (`key`, `value`) of
            unallocated is such that `key` is the variable name, and `value` are
            some hints for the allocation.
        """
        self.statements  = []
        self.unallocated = {}

    @property
    def declarations(self):
        """
        Returns all declarations related to the current statement by looking
        into the global dictionary declarations. the filter is given by
        stmt_vars and local_vars, which must be provided by every extension of
        the base class.
        """
        return [declarations[v] for v in self.stmt_vars + self.local_vars]

    @property
    def local_vars(self):
        """must be defined by the statement."""
        return []

    @property
    def stmt_vars(self):
        """must be defined by the statement."""
        return []

    def update(self):
        pass
