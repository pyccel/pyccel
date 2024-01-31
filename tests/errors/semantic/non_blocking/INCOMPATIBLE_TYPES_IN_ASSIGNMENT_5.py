# pylint: disable=missing-function-docstring, missing-module-docstring

# Not valid in Python 3.8
a : tuple[tuple[int,...], ...] #pylint: disable=unsubscriptable-object
a = ((1,2,3), (4,5,6,7))
