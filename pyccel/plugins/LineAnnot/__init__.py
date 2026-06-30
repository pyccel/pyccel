"""
LineAnnot plugin for pyccel.

When active, this plugin inserts comments in the generated code that map
each statement back to the corresponding line in the original Python source
file, making it easier to trace generated output.
"""
from . import plugin
