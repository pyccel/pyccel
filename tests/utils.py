# coding: utf-8

import os

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
