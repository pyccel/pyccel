# coding: utf-8

from pyccel.parser.syntax.headers import parse

def test_variable():
    print parse(stmts='#$ header variable x :: int')
    print parse(stmts='#$ header variable x float [:, :]')

def test_function():
    print parse(stmts='#$ header function f(float [:], int [:]) results(int)')

def test_class():
    print parse(stmts='#$ header class Square(public)')

def test_method():
    print parse(stmts='#$ header method translate(Point, double, double)')

######################
if __name__ == '__main__':
    test_variable()
    test_function()
    test_class()
    test_method()
