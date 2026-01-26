#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
""" Module containing scripts to manage compilation information.
"""

from .argparse_helpers import add_help_flag, path_with_suffix, add_compiler_selection

__all__ = ('pyccel_config',
           'setup_pyccel_config_parser',
           'PYCCEL_CONFIG_DESCR')

PYCCEL_CONFIG_DESCR = 'Compilation configuration management.'

def setup_pyccel_config_parser(parser):
    """
    Add the `pyccel config` arguments to the parser.

    Add the `pyccel config` arguments to the parser for command line arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to be modified.
    """
    subparsers = parser.add_subparsers(required=True, title='Subcommands', metavar='COMMAND')
    export_parser = subparsers.add_parser('export', add_help=False, help="Export a compiler configuration to a json file.")
    export_parser.add_argument('filename', metavar='FILE', type=path_with_suffix(('.json',), must_exist = False),
                        help='The file that the parser information should be exported to.')

    # ... Compiler options
    add_compiler_selection(export_parser)
    add_help_flag(export_parser.add_argument_group('Options'))

    add_help_flag(parser.add_argument_group('Options'))

def pyccel_config(filename, **kwargs):
    """
    Call the `pyccel config` pipeline.

    Import and call the `pyccel config` pipeline.

    Parameters
    ----------
    filename : Path
        Name of the JSON file where an exported configuration is printed.
    **kwargs : dict
        See execute_pyccel.
    """
    from pyccel.codegen.pipeline  import execute_pyccel
    execute_pyccel('', compiler_export_file = filename, **kwargs)
