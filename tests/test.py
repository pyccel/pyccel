# coding: utf-8

from pyccel.parser  import PyccelParser, get_by_name

# ... creates an instance of Pyccel parser
pyccel = PyccelParser()
# ...

# ...
def test_1():
    # ... parse the Pyccel code
    stmts  = ""
    stmts += "Real :: s" + "\n"

    ast = pyccel.parse(stmts)

    token = get_by_name(ast, "s")
    print token
    # ...
# ...

# ...
def test_2():
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
#    test_1()
    test_2()
