# coding: utf-8

import os
from os.path import join, dirname

from textx.metamodel import metamodel_from_str

#==============================================================================
def parse(inputs, debug=False):
    this_folder = dirname(__file__)

    # Get meta-model from language description
    grammar = join(this_folder, 'grammar.tx')

    from textx.metamodel import metamodel_from_file
    meta = metamodel_from_file(grammar, debug=debug)

    # Instantiate model
    if os.path.isfile(inputs):
        ast = meta.model_from_file(inputs)

    else:
        ast = meta.model_from_str(inputs)

    return ast
