# pylint: disable=missing-function-docstring, missing-module-docstring/
x = 5
y = 0

while y < 3:
    y = y + 1

while y < 3 and x > 2:
    y = y + 1
    x = x - 1
