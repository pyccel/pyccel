# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

from pyccel.parser.syntax.openmp import parse

def test_parallel():
    d = parse(stmts='#$ omp parallel private(idx)')

######################
if __name__ == '__main__':
    test_parallel()
