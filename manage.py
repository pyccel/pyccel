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

def load_extension(ext, module, output_dir, clean=True):
    base_dir = output_dir
    output_dir = path.join(base_dir, ext)
    mkdir_p(output_dir)

    extension = 'pyccelext_{0}'.format(ext)
    try:
        m = importlib.import_module(extension, package=module)
    except:
        raise ImportError('could not import {0}'.format(extension))

    m = getattr(m, '{0}'.format(module))

    # remove 'c' from *.pyc
    filename = m.__file__[:-1]
    print os.path.basename(filename)

    pyccel(files=[filename], output_dir=output_dir)

    # remove .pyccel temporary files
    if clean:
        os.system('rm {0}/*.pyccel'.format(output_dir))

####################################
if __name__ == '__main__':
#    load_extension('math', 'utils', 'extensions')
    load_extension('math', 'quadratures', 'extensions')
