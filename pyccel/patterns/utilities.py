# coding: utf-8

"""
"""

from os.path import join, dirname
from textx.metamodel import metamodel_from_file
from textx.export import metamodel_export, model_export

from pyccel.syntax import ImportFromStmt

def find_imports(filename, debug=False):
    this_folder = dirname(__file__)

    # Get meta-model from language description
    grammar = join(this_folder, 'import.tx')
    meta = metamodel_from_file(grammar, debug=debug, classes=[ImportFromStmt])

    # Instantiate model
    model = meta.model_from_file(filename)

    d = {}
    for stmt in model.statements:
        if isinstance(stmt, ImportFromStmt):
            module = str(stmt.dotted_name.names[0])
            names  = [str(n) for n in stmt.import_as_names.names]
            d[module] = names

    return d
