# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

import os
import pytest

from pyccel.parser.parser   import Parser
from pyccel.codegen.codegen import Codegen
from pyccel.errors.errors   import Errors

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')
failed_tests = ["lists.py"]

files = sorted(os.listdir(path_dir))
files = [
    pytest.param(
        os.path.join(path_dir, f),
        marks=pytest.mark.xfail(reason="Pyccel crash due to list initialization with the use of `len()`, related issue #1924.")
    ) if f in failed_tests else os.path.join(path_dir, f)
    for f in files if f.endswith(".py")
]
@pytest.mark.fortran
@pytest.mark.parametrize("f", files)
def test_codegen(f):

    # reset Errors singleton
    errors = Errors()
    errors.reset()

    pyccel = Parser(f)
    ast = pyccel.parse()

    # Assert syntactic success
    assert not errors.has_errors()

    settings = {}
    ast = pyccel.annotate(**settings)

    # Assert semantic success
    assert not errors.has_errors()

    name = os.path.basename(f)
    name = os.path.splitext(name)[0]

    codegen = Codegen(ast, name, 'fortran')
    codegen.printer.doprint(codegen.ast)

    # Assert codegen success
    assert not errors.has_errors()

######################
if __name__ == '__main__':
    print('*********************************')
    print('***                           ***')
    print('***   TESTING CODEGEN FCODE   ***')
    print('***                           ***')
    print('*********************************')

    for f in files:
        print('> testing {0}'.format(str(os.path.basename(f))))
        test_codegen(f)

    print('\n')
