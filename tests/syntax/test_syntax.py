# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

from pyccel.parser import Parser
from pyccel.parser.errors import Errors
import os

def test_syntax():
    print('*********************************')
    print('***                           ***')
    print('***      TESTING SYNTAX       ***')
    print('***                           ***')
    print('*********************************')

    init_dir = os.getcwd()
    base_dir = os.path.dirname(os.path.realpath(__file__))
    path_dir = os.path.join(base_dir, 'scripts')

    files = sorted(os.listdir(path_dir))
    files = [f for f in files if (f.endswith(".py"))]

    os.chdir(path_dir)
    for f in files:
        print('> testing {0}'.format(str(f)))

        pyccel = Parser(f)
        ast = pyccel.parse()

        # reset Errors singleton
        errors = Errors()
        errors.reset()

    os.chdir(init_dir)
    print('\n')

######################
if __name__ == '__main__':
    test_syntax()
