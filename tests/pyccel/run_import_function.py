# pylint: disable=missing-function-docstring, missing-module-docstring
import sys
import importlib

for modname in sys.argv[1:]:
    mod = importlib.import_module(modname)
    print(mod.test_func())
