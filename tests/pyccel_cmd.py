# coding: utf-8

# if compile is not specified, pyccel will only convert the file to the target language.
# For more info:  python pyccel_cmd.py --help
# Usage :   python pyccel_cmd.py --filename=examples/ex1.py --language="fortran" --compiler="gfortran"


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
                           AssignStmt, \
                           IfStmt, ForStmt, FunctionDefStmt, \
                           ImportFromStmt, \
                           # python standard library statements
                           PythonPrintStmt, \
                           # numpy statments
                           NumpyZerosStmt, NumpyZerosLikeStmt, \
                           NumpyOnesStmt, NumpyLinspaceStmt \
                           )

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
def gencode(ast, printer, name=None, debug=True):
    def gencode_as_module(name, imports, preludes, body):
        # TODO improve if a block is empty
        code  = "module " + str(name)     + "\n"
        code += imports                   + "\n"
        code += "implicit none"           + "\n"
        code += preludes                  + "\n"
        code += "contains"                + "\n"
        code += body                      + "\n"
        code += "end module " + str(name) + "\n"

        return code

    def gencode_as_program(name, imports, preludes, body):
        # TODO improve if a block is empty
        if name is None:
            name = "main"

        code  = "program " + str(name)     + "\n"
        code += imports                   + "\n"
        code += "implicit none"           + "\n"
        code += preludes                  + "\n"
        code += body                      + "\n"
#        code += "contains"                + "\n"
        # TODO add funcdef
        code += "end"                     + "\n"

        return code

    imports  = ""
    preludes = ""
    body    = ""
    for stmt in ast.statements:
        if isinstance(stmt, ImportFromStmt):
            imports += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, DeclarationStmt):
            decs = stmt.expr
            for dec in decs:
                preludes += fcode(dec) + "\n"
        elif isinstance(stmt, NumpyZerosStmt):
            body += fcode(stmt.expr) + "\n"

            for s in stmt.declarations:
                preludes += fcode(s) + "\n"
        elif isinstance(stmt, AssignStmt):
            body += fcode(stmt.expr) + "\n"

            for s in stmt.declarations:
                preludes += fcode(s) + "\n"
        elif isinstance(stmt, ForStmt):
            body += fcode(stmt.expr) + "\n"

            for s in stmt.declarations:
                preludes += fcode(s) + "\n"
        elif isinstance(stmt, IfStmt):
            body += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, FunctionDefStmt):
            body += fcode(stmt.expr) + "\n"+ "\n"
        elif isinstance(stmt, PythonPrintStmt):
            body += fcode(stmt.expr) + "\n"+ "\n"
        else:
            if debug:
                print "> uncovered statement of type : ", type(stmt)
                stmt.expr
            else:
                raise Exception('Statement not yet handled.')

    code = gencode_as_program(name, imports, preludes, body)
#    code = gencode_as_module(name, imports, preludes, body)
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
def compile_file(filename, compiler="gfortran", language="fortran"):
    if not(compiler == "gfortran"):
        raise ValueError("Only gfortran is available")

    if language == "fortran":
        ext = "f90"
    else:
        raise ValueError("Only fortran is available")

    binary = filename.split('.' + ext)[0]
    cmd = compiler + " -O2 " + filename + " -o" + binary
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
parser.add_argument('--execute', action='store_true', \
                    help='executes the binary file')
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
# ...

# ... creates an instance of Pyccel parser
pyccel = PyccelParser()

filename_tmp = make_tmp_file(filename)
preprocess(filename, filename_tmp)

ast = pyccel.parse_from_file(filename_tmp)

name = None
name = "main"
code = gencode(ast, fcode, name=name)

if not execute:
    print "---------------------------"
    print code
    print "---------------------------"
# ...

# ...
filename_out = write_to_file(code, filename, language="fortran")

if compiler:
    binary = compile_file(filename_out, compiler="gfortran", language="fortran")

if compiler and execute:
    execute_file(binary)
# ...
