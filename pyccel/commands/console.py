#!/usr/bin/env python
import sys
import os
import argparse

from pyccel.codegen import build_file

def pyccel():
    """
    pyccel console command.
    """

    class MyParser(argparse.ArgumentParser):
        """
        Custom argument parser for printing help message in case of an error.
        See http://stackoverflow.com/questions/4042452/display-help-message-with-python-argparse-when-script-is-called-without-any-argu
        """
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    parser = MyParser(description='Pyccel command line')
    # ...
#    parser.add_argument('--filename', \
#                        help='config filename. default: config.ini')
    parser.add_argument('--filename', type=str, \
                        help='python file to convert')
    parser.add_argument('--language', type=str, \
                        help='Target language')
    parser.add_argument('--compiler', type=str, \
                        help='Used compiler')
    parser.add_argument('--openmp', action='store_true', \
                        help='uses openmp')
    parser.add_argument('--execute', action='store_true', \
                        help='executes the binary file')
    parser.add_argument('--show', action='store_true', \
                        help='prints the generated file.')
    parser.add_argument('--debug', action='store_true', \
                        help='compiles the code in a debug mode.')
    parser.add_argument('--verbose', action='store_true', \
                        help='enables verbose mode.')
    # ...

    # ...
    args = parser.parse_args()
    # ...

    if args.filename:
        filename = args.filename
    else:
        raise ValueError("a python filename must be provided.")

    if args.language:
        language = args.language
    else:
        raise ValueError("a target language must be provided.")

    if args.compiler:
        compiler = args.compiler
    else:
        compiler = None

    execute = args.execute

    accelerator = None
    if args.openmp:
        accelerator = "openmp"

    debug   = args.debug
    verbose = args.verbose
    show    = args.show
    # ...

    # ...
    build_file(filename, language, compiler, \
            execute=execute, accelerator=accelerator, \
            debug=debug, verbose=verbose, show=show, \
            name="main")
    # ...
