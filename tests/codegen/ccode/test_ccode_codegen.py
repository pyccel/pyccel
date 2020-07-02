# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

import pytest
import os

from pyccel.parser.parser   import Parser
from pyccel.codegen.codegen import Codegen
from pyccel.errors.errors   import Errors

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

failing_files = {'arrays.py':'Arrays not yet supported. See issue 312'}
files = sorted(os.listdir(path_dir))
files = [os.path.join(path_dir,f) \
         if f not in failing_files \
         else pytest.param(os.path.join(path_dir,f), marks = pytest.mark.xfail(reason=failing_files[f])) \
         for f in files \
         if f.endswith(".py") \
        ]
@pytest.mark.c
@pytest.mark.parametrize("f", files)
def test_codegen(f):

    pyccel = Parser(f)
    ast = pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)

    name = os.path.basename(f)
    name = os.path.splitext(name)[0]

    codegen = Codegen(ast, name)
    code = codegen.doprint(language='c')

    # reset Errors singleton
    errors = Errors()
    errors.reset()

######################
if __name__ == '__main__':
    print('*********************************')
    print('***                           ***')
    print('***  TESTING CODEGEN CCODE   ***')
    print('***                           ***')
    print('*********************************')

    for f in files:
        print('> testing {0}'.format(str(os.path.basename(f))))
        codegen_test(f)

    print('\n')

