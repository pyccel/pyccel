
from distutils.core import setup, Extension
 
module = Extension('listModule', sources = ['list_wrapper.c','wrapper_testing.c', '../lists/lists.c'])
 
setup (name = 'listModule',
        version = '',
        description = '',
        ext_modules = [module])