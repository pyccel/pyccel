# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

__all__ = ["BasicStmt"]

class BasicStmt:
    """
    Base class for all objects in Pyccel.

    Conventions:

    1) Every extension class must provide the properties stmt_vars and
    local_vars
    2) stmt_vars describes the list of all variables that are
    created by the statement.
    3) local_vars describes the list of all local variables to the
    statement, like the index of a For statement.
    4) Every extension must implement the update function. This function is
    called to prepare for the applied property (for example the expr
    property.).

    Parameters
    ----------
    **kwargs
        Additional unnecessary arguments provided by textx.
    """

    def __init__(self, **kwargs):
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
