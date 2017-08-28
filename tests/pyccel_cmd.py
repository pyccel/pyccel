# coding: utf-8

# if compile is not specified, pyccel will only convert the file to the target language.
# For more info:  python pyccel_cmd.py --help
# Usage :   python pyccel_cmd.py --language="fortran" --filename=examples/ex1.py
# Usage :   python pyccel_cmd.py --language="fortran" --compiler="gfortran" --filename=examples/ex1.py
# Usage :   python pyccel_cmd.py --language="fortran" --compiler="gfortran" --filename=examples/ex1.py --execute

# using openmp:
#    export OMP_NUM_THREADS=2

import sys
import os
import argparse

from pyccel.codegen import PyccelCodegen, FCodegen
from pyccel.codegen import Compiler, execute_file
from pyccel.codegen import build_file

# ...
parser = argparse.ArgumentParser(description='Pyccel command line.')

#parser.add_argument('--filename', help='config filename. default: config.ini')
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
