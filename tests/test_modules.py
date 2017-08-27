# coding: utf-8

"""
.. todo:
    - no need to declare a variable, if it is defined by assignment. ex: 'x=1'
    means that x is of type double. this must be done automatically.
"""

import sys
import os

from pyccel.codegen import build_file

# ...
build_file(filename="core.py", language="fortran", compiler="gfortran", \
           execute=False, accelerator=None, \
           debug=False, verbose=True, show=False, inline=True, name="core")
# ...

# ...
try:
    import external
    reload(external)
except:
    pass
import external
g = external.core.g
print g(2,3)
# ...
