# pylint: disable=missing-function-docstring, missing-module-docstring
# coding: utf-8

import os
import pytest

from pyccel.parser.parser import Parser
from pyccel.errors.errors import Errors

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]

@pytest.mark.language_agnostic
@pytest.mark.parametrize( "f", files )
def test_semantic(f):

    pyccel = Parser(f, output_folder = os.getcwd())
    pyccel.parse(verbose = 0)

    pyccel.annotate(verbose = 0)

    # reset Errors singleton
    errors = Errors()
    errors.reset()



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

