# The result of a list comprehension expression must be saved in a variable
# pylint: disable=missing-function-docstring, missing-module-docstring
[print(1) for i in range(3)] # pylint: disable=expression-not-assigned
