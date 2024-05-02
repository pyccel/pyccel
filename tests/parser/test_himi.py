# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

from pyccel.parser.syntax.himi import parse

def test_himi_declare():
    print (parse(stmts='E = int'))
    print (parse(stmts='x : int'))


######################
if __name__ == '__main__':
    test_himi_declare()
