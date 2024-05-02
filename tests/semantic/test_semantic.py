# coding: utf-8

from pyccel.parser import Parser
from pyccel.parser.errors import Errors
import os
import pytest

base_dir = os.path.dirname(os.path.realpath(__file__))
path_dir = os.path.join(base_dir, 'scripts')

files = sorted(os.listdir(path_dir))
files = [os.path.join(path_dir,f) for f in files if (f.endswith(".py"))]

@pytest.mark.parametrize( "f", files )
def test_semantic(f):
    print('> testing {0}'.format(str(f)))

    pyccel = Parser(f)
    ast = pyccel.parse()

    settings = {}
    ast = pyccel.annotate(**settings)

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
        test_semantic(f)

    print('\n')

