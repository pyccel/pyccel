#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#

from .argparse_helpers import check_file_type, add_compiler_selection

PYCCEL_CONFIG_DESCR = 'Compilation configuration management.'

def setup_pyccel_config_parser(parser):
    subparsers = parser.add_subparsers(required=True)
    export_parser = subparsers.add_parser('export')
    export_parser.add_argument('filename', metavar='FILE', type=check_file_type(('.json',)),
                        help='The file that the parser information should be exported to.')

    # ... Compiler options
    add_compiler_selection(parser)

def pyccel_config(filename, **kwargs):
    from pyccel.codegen.pipeline  import execute_pyccel
    execute_pyccel('', compiler_export_file = filename, **kwargs)
