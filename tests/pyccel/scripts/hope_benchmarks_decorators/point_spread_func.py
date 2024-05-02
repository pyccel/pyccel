# pylint: disable=missing-function-docstring, missing-module-docstring/
from pyccel.decorators import types

@types('double[:,:]','int','int','double','double','double[:,:]','double','double','double')
def pdf ( density , x_range , y_range , x_center , y_center , w2D, r50 , b , a) :
    from numpy import sqrt, pi, sum as np_sum
    for x in range ( x_range ) :
        for y in range ( y_range ) :
            dr = sqrt ( ( x - x_center ) ** 2 + ( y - y_center ) ** 2)
            tmp = 2 * (b - 1) / (2 * pi * ( r50 * a ) **2) * (1 + ( dr / ( r50 * a ) ) **2)**(-b)
            density [ x , y ] = tmp * np_sum(w2D)

from numpy import zeros

w2D = zeros([7,7])
x_range = 2
y_range = 2
density = zeros([x_range,y_range])
x_center = 5.0
y_center = 5.0
r50 = 50.0
b = 2.0
a = 2.4


rand = 0
rand_a = 100
rand_b = 821
rand_m = 213
for i in range(7):
    for j in range(7):
        rand = (rand_a * rand + rand_b) % rand_m
        w2D[i,j] = float(rand)

pdf ( density , x_range , y_range , x_center , y_center , w2D, r50 , b , a)

for i in range(x_range):
    for j in range(y_range):
        print(density[i,j])
