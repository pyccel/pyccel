
from distutils.core import setup, Extension
 
module = Extension('listModule', sources = ['wrapper_testing.c'])
 
setup (name = 'listModule',
        version = '',
        description = '',
        ext_modules = [module])