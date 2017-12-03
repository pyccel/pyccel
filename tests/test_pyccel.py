# coding: utf-8

import os

from pyccel.commands.console import pyccel
from utils import clean_tests

# ...
def test_1():
    ignored = [15, 18, 20, 21]

    for i in range(1, 21):
        filename = 'tests/scripts/core/ex{0}.py'.format(str(i))
        if not(i in ignored):
            pyccel(files=[filename])
            print(' testing {0}: done'.format(str(i)))
# ...

# ...
def test_2():
    ignored = ['classes.py', 'eval.py', 'parallel.py', 'mpi.py',
              'arrays.py', 'modules.py', 'imports.py', 'dict.py']

    base_dir = os.getcwd()
    path_dir = os.path.join(base_dir, 'tests/scripts')

    files = os.listdir(path_dir)
    files = [f for f in files if not(f in ignored) and (f.endswith(".py"))]

    os.chdir(path_dir)
    for f in files:
        pyccel(files=[f])
        print(' testing {0}: done'.format(str(f)))
    os.chdir(base_dir)
# ...

################################
if __name__ == '__main__':
    clean_tests()
    test_1()
    test_2()
    clean_tests()
