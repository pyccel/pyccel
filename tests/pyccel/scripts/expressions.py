# pylint: disable=missing-function-docstring, missing-module-docstring/
a1 = 2/2
a2 = 2/2j

a3 = 2//2
a4 = 2.//2
a5 = 2//4.

a6 = 2%4
a7 = 2%4.
a8 = 2.%4

a9  = 1.*10
a10 = 1.*10j

b1 = 1-1-2-4-5
b2 = 1-1-2j-4.-5
b3 = 1-1-2j-5
b4 = 1-(1-1)

c1 = 2**2
c2 = 2**2.
c3 = 2**2j
c4 = 2j**3
c5 = 2j**4.
c6 = 2.**-4
c7 = -2.**4
c8 = -.2**-.4

d1 = 1-1-2-4*3+7-5
d2 = 1-1-2**2.-4*3+7-5
d3 = 3*(1+4//2*3.-3.)
d4 = (2*2)**(3*4)

f1  = 100/10/10/2
f2  = 100/-10/10/-2
f3  = 100//4//5//2
f4  = 100//-4//5//-2
f5  = 100.//4//5//2
f6  = 23%24%24%24.
f7  = 23%24%24%24
f8  = 1-2+-2-4-5
f9  = 2*2*-2*-2*3*2
f10 = ++2-+2--2
f11 = 2**3**2
f12 = 2**3j**2

e1 = -(a1 + a3)
e2 = 23*4%16.032/16.4//2./2%3 + 34 + 23*23*4%16.0+32/16.4//2./2%3
e3 = (1%1**4/23**2/6//2.)
e4 = 2 * 4 + 3**4 - 3*5**3//2
e5 = (1%16.032/16.032//25.9948+25.9948*25.99-(25.9-33.74//33.746134%33.746139653844324-33.7464%36.13899562242578)-36.13899562242578*\
     (36.1389-36.138*6.133%6.1332**6.1332%6.1332)%(11.57607+11.5747*11.5760-11.57609-22.9+ 100//6//2.//3)**(0.5))**(3/4 + 3//2)

x = 10
y = 4
g1 = (x==10 or y==10) and (x==20 or y== 20)
g2 = True or True and False
g3 = not True or True and False
g4 = not (True or True and False)
g5 = True and True or False
g6 = x==1 or y == 3 or x == 7 or y == 4
g7 = x==10 and y == 4 and x == 7
g8 = True and True or False and False
g9 = False or True and False or True

print(a1)
print(a2)
print(a3)
print(a4)
print(a5)
print(a6)
print(a7)
print(a8)
print(a9)
print(a10)

print(b1)
print(b2)
print(b3)
print(b4)

print(c1)
print(c2)
print(c3)
print(c4)
print(c5)
print(c6)
print(c7)
print(c8)

print(d1)
print(d2)
print(d3)
print(d4)

print(e1)
print(e2)
print(e3)
print(e4)
print(e5)

print(f1)
print(f2)
print(f3)
print(f4)
print(f5)
print(f6)
print(f7)
print(f8)
print(f9)
print(f10)
print(f11)
print(f12)

print(g1)
print(g2)
print(g3)
print(g4)
print(g5)
print(g6)
print(g7)
print(g8)
print(g9)
