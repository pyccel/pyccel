# coding: utf-8

# Usage:
#   python tests/test_pyccel.py --execute

import os

from pyccel.commands.console import pyccel
from utils import clean_tests

# ...
def test_core():
    print('============== testing core ================')
#    ignored = [15, 18]
    ignored = [15, 18, 5]

    for i in range(0, 23+1):
        filename = 'tests/scripts/core/ex{0}.py'.format(str(i))
        if not(i in ignored):
            pyccel(files=[filename])
            print('> testing {0}: done'.format(str(i)))
# ...

# ...
def test_examples():
    print('============== testing examples ================')

    ignored = ['classes.py', 'eval.py', 'dict.py']

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
def test_oop():
    print('============== testing oop ================')
    ignored = []

    for i in range(0, 4+1):
        filename = 'tests/scripts/oop/ex{0}.py'.format(str(i))
        if not(i in ignored):
            pyccel(files=[filename])
            print('> testing {0}: done'.format(str(i)))
# ...

# ...
def test_lambda():
    print('============== testing lambda ================')
    ignored = []

    base_dir = os.getcwd()

    # ...
    def _get_files(path):
        path_dir = os.path.join(base_dir, path)

        files = sorted(os.listdir(path_dir))
        files = [f for f in files if not(f in ignored) and (f.endswith(".py"))]
        return [os.path.join(path_dir, f) for f in files]
    # ...

    # ...
    folders = ['tests/scripts/lambda/']
    # ...

    # ...
    files = []
    for r in folders:
        files += _get_files(r)
    # ...

    # ...
    for f_name in files:
        f = os.path.basename(f_name)

        pyccel(files=[f_name])
        print('> testing {0}: done'.format(str(f)))
    # ...
# ...

################################
if __name__ == '__main__':
    clean_tests()
    test_core()
    test_examples()
    test_oop()
    test_lambda()
    clean_tests()
