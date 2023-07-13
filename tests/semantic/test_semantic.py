# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

import os
import pytest

from pyccel.parser.parser import Parser
from pyccel.errors.errors import Errors, ErrorsMode

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]

@pytest.mark.parametrize( "f", files )
def test_semantic(f):
    # reset Errors singleton
    errors = Errors()
    errors.reset()
    mode = errors.mode
    ErrorsMode().set_mode('user')

    pyccel = Parser(f)
    pyccel.parse()

    settings = {}
    pyccel.annotate(**settings)

    ErrorsMode().set_mode(mode)



######################
if __name__ == '__main__':
    print('*********************************')
    print('***                           ***')
    print('***      TESTING SEMANTIC     ***')
    print('***                           ***')
    print('*********************************')

    for f in files:
        print('> testing {0}'.format(str(f)))
        test_semantic(f)

    print('\n')

