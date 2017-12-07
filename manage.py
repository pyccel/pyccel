# coding: utf-8

import os
from os import path
import importlib

from pyccel.commands.console import pyccel

def mkdir_p(dir):
    # type: (unicode) -> None
    if path.isdir(dir):
        return
    os.makedirs(dir)

def load_extension(ext, output_dir, clean=True, modules=None, silent=True):

    # ...
    base_dir = output_dir
    output_dir = path.join(base_dir, ext)
    mkdir_p(output_dir)
    # ...

    # ...
    extension = 'pyccelext_{0}'.format(ext)
    try:
        package = importlib.import_module(extension)
    except:
        raise ImportError('could not import {0}'.format(extension))
    ext_dir = str(package.__path__[0])
    # ...

    # ...
    if not modules:
        py_file     = lambda f: (f.split('.')[-1] == 'py')
        ignore_file = lambda f: (os.path.basename(f) in ['__init__.py'])

        files = [f for f in os.listdir(ext_dir) if py_file(f) and not ignore_file(f)]
        modules = [f.split('.')[0] for f in files]
    elif isinstance(modules, str):
        modules = [modules]
    # ...

    for module in modules:
        try:
            m = importlib.import_module(extension, package=module)
        except:
            raise ImportError('could not import {0}.{1}'.format(extension, module))

        m = getattr(m, '{0}'.format(module))

        # remove 'c' from *.pyc
        filename = m.__file__[:-1]

        if not silent:
            print ('> converting {0}/{1}'.format(ext, os.path.basename(filename)))

        pyccel(files=[filename], output_dir=output_dir, compiler=None)

    # remove .pyccel temporary files
    if clean:
        os.system('rm {0}/*.pyccel'.format(output_dir))

####################################
if __name__ == '__main__':
    load_extension('math', 'extensions', silent=False)
    load_extension('math', 'extensions', modules=['bsplines'])
    load_extension('math', 'extensions', modules='quadratures')
