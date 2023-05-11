from distutils.core import setup, Extension

module = Extension('pyccel_append', sources=['lists.c', 'wrapper.c', 'lists_wrapper_tools.c'])

setup(name='pyccel_append', version='1.0', ext_modules=[module])

