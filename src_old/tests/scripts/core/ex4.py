# coding: utf-8

x = 0
for i in range(0,10):
    if i == 5:
        x = x + 1
    else:
        x = 2 * x
    if i != 5:
        x = x + 1
    else:
        x = 2 * x
