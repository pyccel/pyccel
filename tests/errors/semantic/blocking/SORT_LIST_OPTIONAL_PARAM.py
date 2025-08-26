# Optional Parameters are not supported for sort
# pylint: disable=missing-function-docstring, missing-module-docstring

a = [1, 2, -3, -4, 5]
a.sort(reverse=True, key=abs)
