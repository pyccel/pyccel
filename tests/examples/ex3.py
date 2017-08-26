# coding: utf-8

x = 0
y = 0 # TODO not compiled if y is not declared
if x > 1:
    for i in range(0, 4):
        x = x + 1
        y = 3*x
else:
    x = 2 * x
    y = x + 4

x = 0
if x >= 1:
    x = x + 1
else:
    x = 2 * x

x = 0
if x < 1:
    x = x + 1
else:
    x = 2 * x

x = 0
if x <= 1:
    x = x + 1
else:
    x = 2 * x

