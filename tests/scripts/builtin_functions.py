# coding: utf-8

a1 = zeros(64, double)
a2 = zeros(3,2)
a3 = zeros(5, int)
a4 = zeros((2,3), double)

b1 = ones(64, double)
b2 = ones(3,2)
b3 = ones(5, int)
b4 = ones((2,3), double)
b5 = ones((2,3,4), double)

c1 = array((1,2,3,5,8,5),int)
c2 = array(((5,8,6,9,8,2),(5,8,6,9,8,2),(5,8,6,9,8,2),(5,8,6,9,8,2),(5,8,6,9,8,2),(5,8,6,9,8,2)),int)

d0  = abs(-2.0) # TODO fix
d1  = sqrt(2.0)
d2  = sin (2.0)
d3  = cos (2.0)
d4  = tan (2.0)
#d5  = cot (2.0)
d6  = exp (2.0)
d7  = log (2.0)
#d8  = asin(2.0) # TODO need complex numbers
#d9  = acsc(2.0) # TODO need complex numbers
#d10 = acos(2.0) # TODO need complex numbers
#d11 = asec(2.0) # TODO need complex numbers
#d12 = acot(2.0)
d13 = atan(2.0)
#d14 = atan2(2.0)  # TODO must take two args
#d15 = csc (2.0)
#d16 = sec (2.0)
d18 = pow  (2,3)
d19 = sign (-2.0)

e   = 1.0
e0  = 3.0 + 2.0 * abs(e)  # TODO fix
e1  = 3.0 + 2.0 * sqrt(e)
e2  = 3.0 + 2.0 * sin (e)
e3  = 3.0 + 2.0 * cos (e)
e4  = 3.0 + 2.0 * tan (e)
#e5  = 3.0 + 2.0 * cot (e)
e6  = 3.0 + 2.0 * exp (e)
e7  = 3.0 + 2.0 * log (e)
#e8  = 3.0 + 2.0 * asin(e)  # TODO need complex numbers
#e9  = 3.0 + 2.0 * acsc(e)  # TODO need complex numbers
#e10 = 3.0 + 2.0 * acos(e)  # TODO need complex numbers
#e11 = 3.0 + 2.0 * asec(e)  # TODO need complex numbers
#e12 = 3.0 + 2.0 * acot(e)
e13 = 3.0 + 2.0 * atan(e)
#e14 = 3.0 + 2.0 * atan2(e)  # TODO must take two args
#e15 = 3.0 + 2.0 * csc (e)
#e16 = 3.0 + 2.0 * sec (e)
e18 = 3.0 + 2.0 * pow(e, 3)
e19 = 3.0 + 2.0 * sign(e)

n1 = ceil (2.2)
n2 = 3 + 2 * ceil(e)
n3 = len(c1)
n4 = 3 + 2 * len(c1)

#l1, l2 = shape(a4)
#l3, l4, l5 = shape(b5)

r1 = max(a4)
r2 = max(b5)
r3 = min(a4)
r4 = min(b5)

rr1 = 3.0 + 2.0 * max(a4)
rr2 = 3 + 2 * max(b5)
rr3 = 3.0 + 2.0 * min(a4)
rr4 = 3 + 2 * min(b5)
