# Object has already been defined with type 'list[int]'. Type 'tuple[int, ...]' in assignment is incompatible
# pylint: disable=missing-function-docstring, missing-module-docstring

a : list[int,...]
a = (1,2,3)
