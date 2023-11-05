# coding: utf-8
# ------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
# ------------------------------------------------------------------------------------------#
"""
This module defines the parse function used to parse OpenMP comments in Pyccel.
"""

from os.path import join, dirname
from textx.metamodel import metamodel_from_file
from textx.exceptions import TextXError
from pyccel.ast.omp import OmpAnnotatedComment
from .omp_versions import openmp_versions


this_folder = dirname(__file__)
# Get meta-model from language description
grammar = join(this_folder, "../grammar/openmp.tx")

# TODO: get the version from the compiler
omp = openmp_versions["4.5"]
omp_syntax_parser = omp.SyntaxParser
omp_semantic_parser = omp.SemanticParser
omp_ccodeprinter = omp.CCodePrinter
omp.OmpAnnotatedComment.set_current_version(4.5)
omp_classes = omp.inner_classes_list()

meta = metamodel_from_file(grammar, classes=omp_classes)

#def capture_raw_text(omp_statement):
#    from textx.model import get_model
#    the_model = get_model(omp_statement)
#    #raw_str = the_model._tx_parser.parse_tree.value
#
#    #tokens = raw_str.split('|')
#
#    #processed_tokens = [token for token in tokens if token.strip()]
#
#    #omp_statement.statement.raw = ''.join(processed_tokens).replace("#$", "#pragma")
#
#    omp_statement.raw = None

def capture_pos(omp_statement):
    #from textx.model import get_model
    #the_model = get_model(omp_statement)
    omp_statement.pos = (omp_statement._tx_position, omp_statement._tx_position_end)


# object processors: are registered for particular classes (grammar rules)
# and are called when the objects of the given class is instantiated.
# The rules OMP_X_Y are used to insert the version of the syntax used

meta.register_obj_processors({
    'OMP_4_5': lambda _: 4.5,
    'OMP_5_0': lambda _: 5.0,
    'OMP_5_1': lambda _: 5.1,
    'OmpList': capture_pos,
    'OmpIntegerExpr': capture_pos,
    'OmpConstantPositiveInteger': capture_pos,
    'OmpScalarExpr': capture_pos,
})

def parse(stmt, parser, errors):
    """Parse OpenMP code and return a Pyccel AST node.

    Parameters
    ----------
    stmt : str
        The OpenMP code to parse.
    parser : SyntaxParser
        The SyntaxParser object used to parse the code inside the clause.
    errors : Errors
        The errors object to use to report errors.

    Returns
    -------
    Openmp
        The Pyccel AST node representing the OpenMP code.
    """
    try:
        meta_model = meta.model_from_str(stmt)
        #assert len(meta_model.statements) == 1
        meta_model.raw = stmt
        return parser._visit(meta_model.statement)
    except TextXError as e:
        errors.report(e.message, severity="fatal", symbol=stmt)
        return None
