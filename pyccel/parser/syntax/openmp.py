# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/master/LICENSE for full license details.     #
#------------------------------------------------------------------------------------------#
"""
"""

from os.path import join, dirname
from textx.metamodel import metamodel_from_file
from textx.exceptions import TextXError
from .omp_versions import openmp_versions


this_folder = dirname(__file__)
# Get meta-model from language description
grammar = join(this_folder, '../grammar/openmp.tx')

#TODO: get the version from the compiler
used_version = openmp_versions['4.5']
omp_classes = used_version.inner_classes_list()

meta = metamodel_from_file(grammar, classes=omp_classes)

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
        breakpoint()
        meta_model = meta.model_from_str(stmt)
        return meta_model._visit_syntatic(parser, errors)
    except TextXError as e:
        errors.report(e.message, severity='fatal', symbol=stmt)
        return None

