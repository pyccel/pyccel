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

# ...
def clean(filename):
    name = filename.split('.py')[0]
    for ext in ["f90", "pyccel"]:
        f_name = name + "." + ext
        cmd = "rm -f " + f_name
        os.system(cmd)
# ...

# ...
def make_tmp_file(filename):
    name = filename.split('.py')[0]
    return name + ".pyccel"
# ...

# ...
def separator(n=40):
    txt = "."*n
    comment = '!'
    return '{0} {1}\n'.format(comment, txt)
# ...

# ...
def preprocess(filename, filename_out):
    f = open(filename)
    lines = f.readlines()
    f.close()

    # to be sure that we dedent at the end
    lines += "\n"

    lines_new = ""

    def delta(line):
        l = line.lstrip(' ')
        n = len(line) - len(l)
        return n

    tab   = 4
    depth = 0
    for i,line in enumerate(lines):
        n = delta(line)

        if n == depth * tab + tab:
            depth += 1
            lines_new += "indent" + "\n"
            lines_new += line
        else:

            d = n // tab
            if (d > 0) or (n==0):
                old = delta(lines[i-1])
                m = (old - n) // tab
                depth -= m
                for j in range(0, m):
                    lines_new += "dedent" + "\n"

            lines_new += line
    f = open(filename_out, "w")
    for line in lines_new:
        f.write(line)
    f.close()
# ...

# ...
def write_to_file(code, filename, language="fortran"):
    if not(language == "fortran"):
        raise ValueError("Only fortran is available")

    f90_file = filename.split(".py")[0] + ".f90"
    f = open(f90_file, "w")
    for line in code:
        f.write(line)
    f.close()

    return f90_file
# ...

# ...
def compile_file(filename, \
                 compiler="gfortran", language="fortran", \
                 accelerator=None, \
                 debug=False, \
                 verbose=False, \
                 is_module=False, \
                 modules=[]):
    """
    """
    flags = " -O2 "
    if compiler == "gfortran":
        if debug:
            flags += " -fbounds-check "

        if not (accelerator is None):
            if accelerator == "openmp":
                flags += " -fopenmp "
            else:
                raise ValueError("Only openmp is available")
    else:
        raise ValueError("Only gfortran is available")

    if language == "fortran":
        ext = "f90"
    else:
        raise ValueError("Only fortran is available")

#    print "modules : ", modules

    binary = ""
    if not is_module:
        binary = filename.split('.' + ext)[0]
        o_code = " -o "
    else:
        flags += ' -c '
        o_code = ' '
    cmd = compiler + flags + filename + o_code + binary

    if verbose:
        print cmd

    os.system(cmd)

    return binary
# ...

# ...
def execute_file(binary):

    cmd = binary
    if not ('/' in binary):
        cmd = "./" + binary
    os.system(cmd)
# ...

## ...
#try:
#    filename = sys.argv[1]
#except:
#    raise Exception('Expecting a filename')
## ...

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

debug = args.debug
show  = args.show
# ...

# ... creates an instance of Pyccel parser
clean(filename)

filename_tmp = make_tmp_file(filename)
preprocess(filename, filename_tmp)

name = None
name = "main"

codegen = FCodegen(filename=filename_tmp, name=name)
code    = codegen.doprint(language="f95")

is_module = codegen.is_module
modules   = codegen.modules

if is_module:
    execute = False

if show:
    print "---------------------------"
    print code
    print "---------------------------"
# ...

# ...
filename_out = write_to_file(code, filename, language="fortran")

if compiler:
    binary = compile_file(filename_out, \
                          compiler="gfortran", language="fortran", \
                          accelerator=accelerator, \
                          debug=debug, \
                          verbose=False, \
                          is_module=is_module, \
                          modules=modules)

if compiler and execute:
    execute_file(binary)
# ...
