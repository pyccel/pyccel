# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

from pyccel.parser.syntax.openacc import parse

def test_parallel():
    d = parse(stmts='#$ acc parallel private(idx)')

def test_kernels():
    d = parse(stmts='#$ acc kernels')

######################
if __name__ == '__main__':
    test_parallel()
    test_kernels()
