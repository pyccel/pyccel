# coding: utf-8

from pyccel.types.ast   import Import
from pyccel.core.syntax import BasicStmt

__all__ = ['ImportFromStmt']

class ImportFromStmt(BasicStmt):
    """Class representing an Import statement in the grammar."""
    def __init__(self, **kwargs):
        """
        Constructor for an Import statement.

        Parameters
        ==========
        dotted_name: list
            modules path
        import_as_names: textX object
            everything that can be imported
        """
        self.dotted_name     = kwargs.pop('dotted_name')
        self.import_as_names = kwargs.pop('import_as_names')

        super(ImportFromStmt, self).__init__(**kwargs)

    @property
    def expr(self):
        """
        Process the Import statement,
        by returning the appropriate object from pyccel.types.ast
        """
        self.update()

        # TODO how to handle dotted packages?
        fil = self.dotted_name.names[0]
        funcs = self.import_as_names.names
        return Import(fil, funcs)

