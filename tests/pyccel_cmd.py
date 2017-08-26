# coding: utf-8

# if compile is not specified, pyccel will only convert the file to the target language.
# For more info:  python pyccel_cmd.py --help
# Usage :   python pyccel_cmd.py --language="fortran" --filename=examples/ex1.py
# Usage :   python pyccel_cmd.py --language="fortran" --compiler="gfortran" --filename=examples/ex1.py
# Usage :   python pyccel_cmd.py --language="fortran" --compiler="gfortran" --filename=examples/ex1.py --execute

# using openmp:
#    export OMP_NUM_THREADS=2

"""
.. todo:
    - no need to declare a variable, if it is defined by assignment. ex: 'x=1'
    means that x is of type double. this must be done automatically.
"""

import sys
import os
import argparse

from pyccel.printers import fcode
from pyccel.parser  import PyccelParser, get_by_name
from pyccel.syntax import ( \
                           # statements
                           DeclarationStmt, \
                           DelStmt, \
                           PassStmt, \
                           AssignStmt, MultiAssignStmt, \
                           IfStmt, ForStmt,WhileStmt, FunctionDefStmt, \
                           ImportFromStmt, \
                           CommentStmt, AnnotatedStmt, \
                           # python standard library statements
                           PythonPrintStmt, \
                           # numpy statments
                           NumpyZerosStmt, NumpyZerosLikeStmt, \
                           NumpyOnesStmt, NumpyLinspaceStmt,NumpyArrayStmt \

                           )

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
def gencode(filename, printer, name=None, debug=False, accelerator=None):
    # ...
    def gencode_as_module(name, imports, preludes, routines):
        code  = "module " + str(name)     + "\n"
        code += imports                   + "\n"
        code += "implicit none"           + "\n"
        code += preludes                  + "\n"

        if len(routines) > 0:
            code += "contains"            + "\n"
            code += routines              + "\n"
        code += "end module " + str(name) + "\n"

        return code
    # ...

    # ...
    def gencode_as_program(name, imports, preludes, body, routines):
        if name is None:
            name = "main"

        code  = "program " + str(name)    + "\n"
        code += imports                   + "\n"
        code += "implicit none"           + "\n"
        code += preludes                  + "\n"

        if len(body) > 0:
            code += body                  + "\n"
        if len(routines) > 0:
            code += "contains"            + "\n"
            code += routines              + "\n"
        code += "end"                     + "\n"

        return code
    # ...

    # ...
    pyccel = PyccelParser()
    ast = pyccel.parse_from_file(filename)
    # ...

    # ...
    imports  = ""
    preludes = ""
    body     = ""
    routines = ""
    # ...

    # ... TODO improve. mv somewhere else
    if not (accelerator is None):
        if accelerator == "openmp":
            imports += "use omp_lib "
        else:
            raise ValueError("Only openmp is available")
    # ...

    # ...
    for stmt in ast.statements:
        if isinstance(stmt, CommentStmt):
            body += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, AnnotatedStmt):
            body += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, ImportFromStmt):
            imports += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, DeclarationStmt):
            decs = stmt.expr
        elif isinstance(stmt, NumpyZerosStmt):
            body += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, NumpyZerosLikeStmt):
            body += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, NumpyOnesStmt):
            body += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, NumpyLinspaceStmt):
            body += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, NumpyArrayStmt):
            body += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, AssignStmt):
            body += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, MultiAssignStmt):
            body += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, ForStmt):
            body += fcode(stmt.expr) + "\n"
        elif isinstance(stmt,WhileStmt):
            body += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, IfStmt):
            body += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, FunctionDefStmt):
            routines += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, PythonPrintStmt):
            body += fcode(stmt.expr) + "\n"
        else:
            if debug:
                print "> uncovered statement of type : ", type(stmt)
            else:

                raise Exception('Statement not yet handled.')
    # ...

    # ...
    for key, dec in ast.declarations.items():
        preludes += fcode(dec) + "\n"
    # ...

    code = gencode_as_program(name, imports, preludes, body, routines)
#    code = gencode_as_module(name, imports, preludes, routines)
    return code
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
                 verbose=False):
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

    binary = filename.split('.' + ext)[0]
    cmd = compiler + flags + filename + " -o" + binary

    if verbose:
        print cmd

    os.system(cmd)

    return binary
# ...

# ...
def execute_file(binary):

    cmd = binary
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
code = gencode(filename_tmp, fcode, \
               name=name, \
               accelerator=accelerator)

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
                          verbose=False)

if compiler and execute:
    execute_file(binary)
# ...
