# coding: utf-8
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
    parser.add_argument('--analysis', action='store_true', \
                        help='enables code analysis mode.')
    parser.add_argument('--local_vars', type=str, \
                        help='local variables on fast memory.')
    # TODO: remove local_vars later, by using Annotated Comments
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
        if not args.analysis:
            raise ValueError("a target language must be provided.")

    if args.compiler:
        compiler = args.compiler
    else:
        compiler = None

    execute = args.execute

    accelerator = None
    if args.openmp:
        accelerator = "openmp"

    debug      = args.debug
    verbose    = args.verbose
    show       = args.show
    analysis   = args.analysis
    # ...

    # ...
    if not analysis:
        build_file(filename, language, compiler, \
                execute=execute, accelerator=accelerator, \
                debug=debug, verbose=verbose, show=show, \
                name="main")
    else:
        from pyccel.complexity.memory import MemComplexity

        local_vars = []
        if args.local_vars:
            local_vars = args.local_vars.split(',')
        complexity = MemComplexity(filename)
        complexity.intensity(verbose=True, local_vars=local_vars)
    # ...
