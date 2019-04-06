# coding: utf-8

import os
from os.path import join, dirname

from textx.metamodel import metamodel_from_str


#==============================================================================
class NamedAbstraction(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')

class Abstraction(object):
    def __init__(self, **kwargs):
        self.args = kwargs.pop('args')
        self.expr = kwargs.pop('expr')

class Application(object):
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.args = kwargs.pop('args')

#==============================================================================
def parse(inputs, debug=False):
    this_folder = dirname(__file__)

    classes = [NamedAbstraction, Abstraction, Application]

    # Get meta-model from language description
    grammar = join(this_folder, 'grammar.tx')

    from textx.metamodel import metamodel_from_file
    meta = metamodel_from_file(grammar, debug=debug, classes=classes)

    # Instantiate model
    if os.path.isfile(inputs):
        ast = meta.model_from_file(inputs)

    else:
        ast = meta.model_from_str(inputs)

    return ast
