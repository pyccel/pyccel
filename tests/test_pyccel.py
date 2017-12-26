# coding: utf-8

# Usage:
#   python tests/test_pyccel.py --execute

import os

from pyccel.commands.console import pyccel
from utils import clean_tests

# ...
def test_1():
#    ignored = [15, 18]
    ignored = [15, 18, 5, 21, 22]

    for i in range(0, 23+1):
        filename = 'tests/scripts/core/ex{0}.py'.format(str(i))
        if not(i in ignored):
            pyccel(files=[filename])
            print('> testing {0}: done'.format(str(i)))
# ...

# ...
def test_2():
#    ignored = ['eval.py', 'parallel.py', 'mpi.py',
#               'modules.py', 'imports.py', 'dict.py']

    ignored = ['functions.py', 'classes.py', 'arrays.py',
               'eval.py', 'parallel.py', 'mpi.py',
               'modules.py', 'imports.py', 'dict.py']

    base_dir = os.getcwd()
    path_dir = os.path.join(base_dir, 'tests/scripts')

    files = sorted(os.listdir(path_dir))
    files = [f for f in files if not(f in ignored) and (f.endswith(".py"))]

    for f in files:
        print(' {0}: '.format(str(f)))
        f_name = os.path.join(path_dir, f)

        pyccel(files=[f_name])
        print('> testing {0}: done'.format(str(f)))
# ...

# ...
def test_3():
    ignored = []

    for i in range(0, 1+1):
        filename = 'tests/scripts/oop/ex{0}.py'.format(str(i))
        if not(i in ignored):
            pyccel(files=[filename])
            print('> testing {0}: done'.format(str(i)))
# ...

################################
if __name__ == '__main__':
    clean_tests()
    test_1()
    test_2()
    test_3()
    clean_tests()
