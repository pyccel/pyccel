# coding: utf-8

"""
"""

from pyccel.patterns.utilities import find_imports

f = 'ex_import.py'
f = 'ex6.py'
imports = find_imports(filename=f)
print(imports)
