# coding: utf-8

# Usage:
#   python tests/test_pyccel_lapack.py --libs='blas lapack' --execute

import os

from pyccel.commands.console import pyccel
from utils import clean_tests

# ...
def test_blaslapack():
    ignored = []

    base_dir = os.getcwd()
    path_dir = os.path.join(base_dir, 'tests/scripts/lapack')

    files = sorted(os.listdir(path_dir))
    files = [f for f in files if not(f in ignored) and (f.endswith(".py"))]

    for f in files:
        f_name = os.path.join(path_dir, f)

        pyccel(files=[f_name])
        print('> testing {0}: done'.format(str(f)))
# ...

################################
if __name__ == '__main__':
    clean_tests()
    test_blaslapack()
    clean_tests()
