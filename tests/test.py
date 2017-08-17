# coding: utf-8

from pyccel.parser  import PyccelParser, get_by_name
from pyccel.syntax import ( \
                           # statements
                           DeclarationStmt, \
                           DelStmt, \
                           PassStmt, \
                           AssignStmt, \
                           ForStmt)

# ... creates an instance of Pyccel parser
pyccel = PyccelParser()
# ...

# ...
def test_Declare():
    # ... parse the Pyccel code
    stmts  = ""
    stmts += "real x,y" + "\n"
    stmts += "real z"   + "\n"
    stmts += "int  n"   + "\n"

    ast = pyccel.parse(stmts)

    for stmt in ast.statements:
        if isinstance(stmt, DeclarationStmt):
            print stmt.variables
    # ...
# ...

# ...
def test_Pass():
    # ... parse the Pyccel code
    stmts  = ""
    stmts += "pass" + "\n"

    ast = pyccel.parse(stmts)
    for stmt in ast.statements:
        print type(stmt)
    # ...
# ...

# ...
def test_Flow():
    # ... parse the Pyccel code
    stmts  = ""
    stmts += "return"   + "\n"
    stmts += "raise"    + "\n"
    stmts += "break"    + "\n"
    stmts += "continue" + "\n"

    ast = pyccel.parse(stmts)
    for stmt in ast.statements:
        print type(stmt)
    # ...
# ...

# ...
def test_Del():
    # ... parse the Pyccel code
    stmts  = ""
    stmts += "real b" + "\n"
    stmts += "del b"   + "\n"

    ast = pyccel.parse(stmts)
    for stmt in ast.statements:
        print stmt.expr
    # ...
# ...

# ...
def test_Assign():
    # ... parse the Pyccel code
    stmts  = ""
    stmts += "real  a,b" + "\n"
    stmts += "x=1"       + "\n"
    stmts += "y=2*3+1"   + "\n"
    stmts += "x=a"       + "\n"
    stmts += "y=2*a+b"   + "\n"

    ast = pyccel.parse(stmts)
    for stmt in ast.statements:
        if isinstance(stmt, DeclarationStmt):
            print "declared variable : ", stmt.variables
        if isinstance(stmt, AssignStmt):
            print "lhs : ", stmt.lhs, "     rhs: ", stmt.rhs.expr
    # ...
# ...

# ...
def test_If():
    # ... parse the Pyccel code
    stmts  = ""
    stmts += "real  a,b" + "\n"
    stmts += "if a==1:"       + "\n"
    stmts += "b=a+1"   + "\n"
#    stmts += ";"   + "\n"

    ast = pyccel.parse(stmts)
    for stmt in ast.statements:
        print type(stmt)
#        if isinstance(stmt, DeclarationStmt):
#            print "declared variable : ", stmt.variables
#        if isinstance(stmt, AssignStmt):
#            print "lhs : ", stmt.lhs, "     rhs: ", stmt.rhs.expr
    # ...
# ...

# ...
def test_For():
    # ... parse the Pyccel code
    stmts  = ""
    stmts += "for i in range(0,10):" + "\n"
    stmts += "x=1"                + "\n"
    stmts += "end"                   + "\n"

    stmts += "for j in range(a,b):" + "\n"
    stmts += "x=1"                + "\n"
    stmts += "end"                   + "\n"

    ast = pyccel.parse(stmts)
    for stmt in ast.statements:
        print stmt.expr

#    token = get_by_name(ast, "s")
#    print token
    # ...
# ...


######################################
if __name__ == "__main__":
#    test_Assign()
#    test_Declare()
#    test_Del()
#    test_Flow()
    test_If()
#    test_For() # KO
#    test_Pass()
