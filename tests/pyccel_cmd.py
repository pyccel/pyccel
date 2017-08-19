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
                           IfStmt, ForStmt)

# ...
def gencode(ast, printer):
    preludes = ""
    lines    = ""
    for stmt in ast.statements:
        if isinstance(stmt, DeclarationStmt):
            decs = stmt.expr
            for dec in decs:
                preludes += fcode(dec) + "\n"
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
        else:
            raise Exception('Statement not yet handled.')

    code = preludes + "\n" + lines
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
ast = pyccel.parse_from_file(filename)
code = gencode(ast, fcode)
# ...

# ...
print code
# ...
