# pylint: disable=missing-function-docstring, missing-module-docstring
x = 5
y = 0

while y < 3:
    y = y + 1

while y < 3 and x > 2:
    y = y + 1
    x = x - 1

while y < 3 and x > 2 and x > y:
    y = y + 1
    x = x - 1

while (y < 3 and x > 2) or x > y:
    y = y + 1
    x = x - 1

while x:
    x = x - 1

while not(x):
    x = x + 1

while not x or x < 4:
    x = x + 1
