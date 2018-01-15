# coding: utf-8

# Usage:
#   python tests/test_pyccel_openmp.py --openmp --execute

import os

from pyccel.commands.console import pyccel
from utils import clean_tests

# ...
def test_core():
    print('============== testing core ================')
    ignored = []

    base_dir = os.getcwd()
    path_dir = os.path.join(base_dir, 'tests/scripts/openmp/core')

    files = sorted(os.listdir(path_dir))
    files = [f for f in files if not(f in ignored) and (f.endswith(".py"))]

    for f in files:
        f_name = os.path.join(path_dir, f)

        pyccel(files=[f_name], openmp=True)
        print('> testing {0}: done'.format(str(f)))
# ...

# ...
def test_openmp():
    print('============== testing examples ================')
    ignored = ['matrix_multiplication.py']

    base_dir = os.getcwd()
    path_dir = os.path.join(base_dir, 'tests/scripts/openmp')

    files = sorted(os.listdir(path_dir))
    files = [f for f in files if not(f in ignored) and (f.endswith(".py"))]

    for f in files:
        f_name = os.path.join(path_dir, f)

        pyccel(files=[f_name], openmp=True)
        print('> testing {0}: done'.format(str(f)))
# ...

################################
if __name__ == '__main__':
    clean_tests()
    test_core()
    test_openmp()
    clean_tests()
