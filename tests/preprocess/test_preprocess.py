# coding: utf-8

# Note that we need to change the directory for tests involving the import
# statement

from pyccel.parser.errors import Errors
from pyccel.parser import Parser
import os

def test_preprocess():
    print('*********************************')
    print('***                           ***')
    print('***     TESTING preprocess    ***')
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
        pyccel.parse()
        print(pyccel.fst)

        # reset Errors singleton
        errors = Errors()
        errors.reset()

    os.chdir(init_dir)
    print('\n')

######################
if __name__ == '__main__':
    test_preprocess()
