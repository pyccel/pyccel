# pylint: disable=missing-function-docstring, missing-module-docstring/
from numpy import zeros

x = 0
y = 0
z = 3
b = zeros(64,'int')

if 1<0:
    x=4

if len(b)>5:
    x = len(b)

if (x==2 or y==1) and z>3 :
    x = 3
    y = 4


if x>2 or y<=1:
    x = x-1
    y = y-1


if True:
    x = 5

if False:
    y = 0

if x > 1:
    for i in range(0,4):
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
if x==0:
    x = x + 1
else:
    x = 2 * x


x = 0
if x == 1 and x>2:
    x = x + 1
else:
    y = 2 * x

x = 0
if x != 1:
    x = x + 1
else:
    y = 2 * x


x = 0
if x > 5:
    x = x + 5
    x = x - 5
elif x > 4:
    x = x + 4
    x = x - 4
elif x > 3:
    x = x + 3
    x = x - 3
else:
    x = 2 * x

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
