# coding: utf-8

from pyccel.parser  import PyccelParser, get_by_name

# ... creates an instance of Pyccel parser
pyccel = PyccelParser()
# ...

# ...
def test_Declare():
    # ... parse the Pyccel code
    stmts  = ""
    stmts += "real x;" + "\n"
    stmts += "int  n;" + "\n"

    ast = pyccel.parse(stmts)

    for t in ["x", "n"]:
        token = get_by_name(ast, t)
        print token.expr, " of type ", token.datatype
    # ...
# ...

# ...
def test_Assign():
    # ... parse the Pyccel code
    stmts  = ""
    stmts += "real  a;" + "\n"  # KO
#    stmts += "real :: b" + "\n"  # KO
    stmts += "x=1"       + "\n"  # OK
#    stmts += "y=2*3+1"   + "\n"  # OK
#    stmts += "x=a"       + "\n"  # OK
#    stmts += "y=2*a+b"   + "\n"   # KO

    ast = pyccel.parse(stmts)
    for stmt in ast.declarations:
        print stmt.expr
    for stmt in ast.statements:
        print stmt.expr
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
    test_Declare()
#    test_Assign()
#    test_For()
