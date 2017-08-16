# coding: utf-8

from pyccel.parser  import PyccelParser, get_by_name

# ... creates an instance of Pyccel parser
pyccel = PyccelParser()
# ...

# ...
def test_1():
    # ... parse the Pyccel code
    stmts  = ""
    stmts += "Real                            :: s"     + "\n"

    ast = pyccel.parse(stmts)

    token = get_by_name(ast, "s")
    print token
    # ...

# ...


######################################
if __name__ == "__main__":
    test_1()
