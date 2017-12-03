# coding: utf-8

import os

from pyccel.commands.console import pyccel

# ...
def clean_tests():
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk("."):
        path = root.split(os.sep)
        dirname = '/'.join(i for i in path)
        _files = []
        for f in files:
            ext = f.split('.')[-1]
            name = f.split('.')[0]
            if (ext == 'py') and not(name in ['__init__']):
                _files.append(name)
        for f in _files:
            f_name = '{0}/{1}'.format(dirname, f)
            if os.path.isfile(f_name):
                cmd = 'rm -f '+f_name
                os.system(cmd)
                for ext in ['f90', 'pyccel']:
                    cmd = 'rm -f {0}.{1}'.format(f_name, ext)
                    os.system(cmd)
# ...

# ...
def test_1():
    ignored = [11, 15, 18, 20, 21]

    for i in range(1, 21):
        filename = 'tests/scripts/core/ex{0}.py'.format(str(i))
        if not(i in ignored):
            pyccel(files=[filename])
            print(' testing {0}: done'.format(str(i)))
# ...

# ...
def test_2():
    ignored = ['classes.py', 'eval.py', 'parallel.py', 'mpi.py',
              'arrays.py', 'modules.py', 'imports.py']

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
