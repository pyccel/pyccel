# coding: utf-8

"""
.. todo:
    - no need to declare a variable, if it is defined by assignment. ex: 'x=1'
    means that x is of type double. this must be done automatically.
"""

import sys
import os
import argparse

from pyccel.codegen import PyccelCodegen, FCodegen
from pyccel.codegen import Compiler, execute_file

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

# ... creates an instance of Pyccel parser
name = None
name = "main"

codegen_m = FCodegen(filename="core.py", name="core", is_module=True)
codegen_m.doprint(language="fortran")
print ">>> Codegen : core done."

codegen = FCodegen(filename=filename, name=name)
codegen.doprint(language="fortran")
print ">>> Codegen :", name, " done."

modules   = codegen.modules
# ...

# ...
if compiler:
    compiler_m = Compiler(codegen_m, compiler="gfortran", debug=debug)
    compiler_m.compile(verbose=verbose)

    compiler = Compiler(codegen, compiler="gfortran", debug=debug)
    compiler.compile(verbose=verbose)
    binary   = compiler.binary

    if execute:
        execute_file(binary)
# ...
