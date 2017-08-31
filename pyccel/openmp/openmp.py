# coding: utf-8

"""
"""

from os.path import join, dirname
from textx.metamodel import metamodel_from_file
from textx.export import metamodel_export, model_export

from pyccel.syntax import BasicStmt

class ParallelNumThreadClause(BasicStmt):
    """Class representing a ."""
    def __init__(self, **kwargs):
        """
        """
        self.clause = kwargs.pop('clause')
        self.thread = kwargs.pop('thread')

        super(ParallelNumThreadClause, self).__init__(**kwargs)

    @property
    def expr(self):
        pass

def parse(filename, debug=False):
    this_folder = dirname(__file__)

    # Get meta-model from language description
    grammar = join(this_folder, 'openmp.tx')
    meta = metamodel_from_file(grammar, debug=debug, \
                               classes=[ParallelNumThreadClause])
#    meta = metamodel_from_file(grammar, debug=debug)

    # Instantiate model
    model = meta.model_from_file(filename)

    d = {}
    for stmt in model.statements:
        print type(stmt), stmt
#        if isinstance(stmt, ImportFromStmt):
#            module = str(stmt.dotted_name.names[0])
#            names  = [str(n) for n in stmt.import_as_names.names]
#            d[module] = names

    return d

####################################
if __name__ == '__main__':
    filename = 'test.py'
    parse(filename, debug=False)
