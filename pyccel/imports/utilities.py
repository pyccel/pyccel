# coding: utf-8

from os.path import join, dirname
from textx.metamodel import metamodel_from_file
from textx.export import metamodel_export, model_export

from pyccel.imports.syntax import ImportFromStmt

__all__ = ['find_imports']

def find_imports(filename, debug=False):
    """
    Finds all import statements in the file.

    filename: str
        name of the file to parse
    debug: bool
        use debug mode if True
    """
    this_folder = dirname(__file__)

    # Get meta-model from language description
    grammar = join(this_folder, 'grammar.tx')
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
