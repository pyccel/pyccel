# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

from pyccel.parser.syntax.headers import parse

def test_variable():
    print (parse(stmts='#$ header variable x :: int'))
    print (parse(stmts='#$ header variable x float [:, :]'))

def test_function():
    print (parse(stmts='#$ header function f(float [:], int [:]) results(int)'))

def test_function_static():
    print (parse(stmts='#$ header function static f(float [:], int [:]) results(int)'))

def test_class():
    print (parse(stmts='#$ header class Square(public)'))

def test_method():
    print (parse(stmts='#$ header method translate(Point, double, double)'))

def test_metavar():
    print (parse(stmts="#$ header metavar module_name='mpi'"))

def test_macro():
    print (parse(stmts='#$ header macro _f(x) := f(x, x.shape)'))
    print (parse(stmts='#$ header macro _g(x) := g(x, x.shape[0], x.shape[1])'))
    print (parse(stmts='#$ header macro (a, b), _f(x) := f(x.shape, x, a, b)'))
    print (parse(stmts='#$ header macro _dswap(x, incx) := dswap(x.shape, x, incx)'))
    print (parse(stmts="#$ header macro _dswap(x, incx=1) := dswap(x.shape, x, incx)"))
    print (parse(stmts='#$ header macro _dswap(x, y, incx=1, incy=1) := dswap(x.shape, x, incx, y, incy)'))

######################
if __name__ == '__main__':
    test_variable()
    test_function()
    test_function_static()
    test_class()
    test_method()
    test_metavar()
    test_macro()
