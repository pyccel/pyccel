# pylint: disable=missing-function-docstring, missing-module-docstring/
# coding: utf-8

import os
import pytest

from pyccel.parser.parser   import Parser
from pyccel.codegen.codegen import Codegen
from pyccel.errors.errors   import Errors

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [f for f in files if (f.endswith(".py"))]

@pytest.mark.parametrize( "f", files )
@pytest.mark.xfail(reason="Broken symbolic function support, see issue #330")
def test_symbolic(f):

    pyccel = Parser(f)
    pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)

    name = os.path.basename(f)
    name = os.path.splitext(name)[0]

    codegen = Codegen(ast, name)
    codegen.doprint()

    # reset Errors singleton
    errors = Errors()
    errors.reset()

######################
if __name__ == '__main__':
    print('*********************************')
    print('***                           ***')
    print('***      TESTING SYMBOLIC     ***')
    print('***                           ***')
    print('*********************************')

    for f in files:
        print('> testing {0}'.format(str(f)))
        test_symbolic(f)
        print('\n')
