# coding: utf-8

"""
"""

from os.path import join, dirname
from textx.metamodel import metamodel_from_file
from textx.export import metamodel_export, model_export


def main(debug=False):

    this_folder = dirname(__file__)

    # Get meta-model from language description
    meta = metamodel_from_file(join(this_folder, 'import.tx'), debug=debug)

    # Instantiate model
    model = meta.model_from_file(join(this_folder, 'ex_import.py'))

    print model.statements

if __name__ == '__main__':
    main()




