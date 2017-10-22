# coding: utf-8

from os.path import join, dirname
from textx.metamodel import metamodel_from_file
from textx.export import metamodel_export, model_export

from pyccel.parser.syntax.core import ImportFromStmt, ImportAsNames
from pyccel.parser.syntax.core import clean_namespace

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
    grammar = join(this_folder, 'grammar/imports.tx')
    meta = metamodel_from_file(grammar, debug=debug, \
                               classes=[ImportFromStmt, ImportAsNames])

    # Instantiate model
    model = meta.model_from_file(filename)

    d = {}
    for stmt in model.statements:
        if isinstance(stmt, ImportFromStmt):
            expr = stmt.expr
            module = str(expr.fil)
            names  = str(expr.funcs)
            d[module] = names

    clean_namespace()

    return d
