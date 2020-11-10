# pylint: disable=missing-function-docstring, missing-module-docstring/
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

files = sorted(os.listdir(path_dir))
files = [os.path.join(path_dir,f) \
         #if f not in failing_files \
         #else pytest.param(os.path.join(path_dir,f), marks = pytest.mark.xfail(reason=failing_files[f])) \
         for f in files \
         if f.endswith(".py") \
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
    assert(not errors.has_errors())

    settings = {}
    ast = pyccel.annotate(**settings)

    # Assert semantic success
    assert(not errors.has_errors())

    name = os.path.basename(f)
    name = os.path.splitext(name)[0]

    codegen = Codegen(ast, name)
    codegen.doprint()

    # Assert codegen success
    assert(not errors.has_errors())

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
