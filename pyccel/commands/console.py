#!/usr/bin/env python
# coding: utf-8
#------------------------------------------------------------------------------------------#
# This file is part of Pyccel which is released under MIT License. See the LICENSE file or #
# go to https://github.com/pyccel/pyccel/blob/devel/LICENSE for full license details.      #
#------------------------------------------------------------------------------------------#
import argparse
import pathlib
import sys

from .pyccel_clean import setup_pyccel_clean_parser, pyccel_clean_loop, PYCCEL_CLEAN_DESCR
from .pyccel_compile import setup_pyccel_compile_parser, pyccel, PYCCEL_COMPILE_DESCR
from .pyccel_make import setup_pyccel_make_parser, pyccel_make, PYCCEL_MAKE_DESCR
from .pyccel_test import setup_pyccel_test_parser, pyccel_test, PYCCEL_TEST_DESCR
from .pyccel_wrap import setup_pyccel_wrap_parser, pyccel_wrap, PYCCEL_WRAP_DESCR
from .pyccel_config import setup_pyccel_config_parser, pyccel_config, PYCCEL_CONFIG_DESCR
from .argparse_helpers import add_version_flag

def pyccel_command() -> None:
    """
    Pyccel console command.

    The Pyccel console command allows translating Python files using Pyccel in
    a command-line environment. This function takes no parameters and sets up
    an argument parser for the Pyccel command line interface.

    The command line interface requires a Python file to be specified, and it
    supports various options such as specifying the output language (C,
    Fortran, or Python), compiler settings, and flags for accelerators like
    MPI, OpenMP, and OpenACC. It also includes options for verbosity,
    debugging, and exporting compile information. Unless the user requires the
    process to stop after a specific stage, Pyccel will execute the full
    translation and compilation process until a C Python extension module is
    generated, which can then be imported in Python. In addition, if the input
    file contains an `if __name__ == '__main__':` block, an executable will be
    generated for the corresponding block of code.
    """
    parser = argparse.ArgumentParser(description="Pyccel's command line interface.",
                      add_help=True, exit_on_error=False)

    #... Help and Version
    add_version_flag(parser)

    sub_commands = {'clean': (setup_pyccel_clean_parser, pyccel_clean_loop, PYCCEL_CLEAN_DESCR),
                    'compile' : (setup_pyccel_compile_parser, pyccel, PYCCEL_COMPILE_DESCR),
                    'config': (setup_pyccel_config_parser, pyccel_config, PYCCEL_CONFIG_DESCR),
                    'make':  (setup_pyccel_make_parser, pyccel_make, PYCCEL_MAKE_DESCR),
                    'test':  (setup_pyccel_test_parser, pyccel_test, PYCCEL_TEST_DESCR),
                    'wrap':  (setup_pyccel_wrap_parser, pyccel_wrap, PYCCEL_WRAP_DESCR),
                    }

    subparsers = parser.add_subparsers(required=True)
    for key, (parser_setup, exe_func, descr) in sub_commands.items():
        sparser = subparsers.add_parser(key, help=descr)
        parser_setup(sparser)
        sparser.set_defaults(func=exe_func)

    argv = sys.argv[1:]
    if len(argv) == 0:
        parser.print_help()
        sys.exit(2)

    try:
        kwargs = vars(parser.parse_args())
    except argparse.ArgumentError:
        print("warning: Using pyccel with no sub-command is deprecated and will be removed in v2.3. Please use pyccel compile.",
              file=sys.stderr)
        argv = ('compile', *argv)
        parser.exit_on_error=True
        kwargs = vars(parser.parse_args(argv))

    from pyccel.errors.errors     import PyccelError, Errors
    from pyccel.utilities.stage   import PyccelStage

    pyccel_stage = PyccelStage()
    errors = Errors()

    func = kwargs.pop('func')
    try:
        func(**kwargs)
    except PyccelError:
        pass

    pyccel_stage.pyccel_finished()
    print(errors, end='')
    sys.exit(errors.has_errors())
