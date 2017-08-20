# coding: utf-8

"""
.. todo:
    - no need to declare a variable, if it is defined by assignment. ex: 'x=1'
    means that x is of type double. this must be done automatically.
"""

import sys
import os
from symcc.printers import fcode

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
def gencode(ast, printer):
    imports  = ""
    preludes = ""
    lines    = ""
    for stmt in ast.statements:
        if isinstance(stmt, ImportFromStmt):
            imports += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, DeclarationStmt):
            decs = stmt.expr
            for dec in decs:
                preludes += fcode(dec) + "\n"
        elif isinstance(stmt, NumpyZerosStmt):
            lines += fcode(stmt.expr) + "\n"

            for s in stmt.statements:
                preludes += fcode(s) + "\n"
        elif isinstance(stmt, AssignStmt):
            lines += fcode(stmt.expr) + "\n"

            for s in stmt.statements:
                preludes += fcode(s) + "\n"
        elif isinstance(stmt, ForStmt):
            lines += fcode(stmt.expr) + "\n"

            for s in stmt.statements:
                preludes += fcode(s) + "\n"
        elif isinstance(stmt, IfStmt):
            lines += fcode(stmt.expr) + "\n"
        elif isinstance(stmt, FunctionDefStmt):
            lines += fcode(stmt.expr) + "\n"+ "\n"
        else:
            raise Exception('Statement not yet handled.')

    code = imports + "\n"  \
         + preludes + "\n" \
         + lines

    return code
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
code = gencode(ast, fcode)

f90_file = filename.split(".py")[0] + ".f90"
f = open(f90_file, "w")
for line in code:
    f.write(line)
f.close()
# ...

# ...
print code
# ...
