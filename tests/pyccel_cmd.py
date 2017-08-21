# coding: utf-8

"""
.. todo:
    - no need to declare a variable, if it is defined by assignment. ex: 'x=1'
    means that x is of type double. this must be done automatically.
"""

import sys
import os
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
def gencode(ast, printer, name=None):
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
# ...

# ...
try:
    filename = sys.argv[1]
except:
    raise Exception('Expecting a filename')
# ...

# ... creates an instance of Pyccel parser
pyccel = PyccelParser()

filename_tmp = make_tmp_file(filename)
preprocess(filename, filename_tmp)

ast = pyccel.parse_from_file(filename_tmp)

name = None
name = "main"
code = gencode(ast, fcode, name=name)

print code
# ...

# ...
filename_out = write_to_file(code, filename, language="fortran")
compile_file(filename_out, compiler="gfortran", language="fortran")
# ...
