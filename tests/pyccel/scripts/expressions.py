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

e1 = -(a1 + a3)
e2 = 23*4%16.032/16.4//2./2%3 + 34 + 23*23*4%16.0+32/16.4//2./2%3
e3 = (1%1**4/23**2/6//2.)
e4 = 2 * 4 + 3**4 - 3*5**3//2
e5 = (1%16.032/16.032//25.9948+25.9948*25.99-(25.9-33.74//33.746134%33.746139653844324-33.7464%36.13899562242578)-36.13899562242578*\
     (36.1389-36.138*6.133%6.1332**6.1332%6.1332)%(11.57607+11.5747*11.5760-11.57609-22.9+ 100//6//2.//3)**(0.5))**(3/4 + 3//2)

print(a1==1.0)
print(a2==-1j)
print(a3==1.)
print(a4==1.)
print(a5==0.)
print(a6==2.)
print(a7==2.)
print(a8==2.)
print(a9==10.)
print(a10==10j)

print(b1==-11)
print(b2==-9-2j)
print(b3==-5-2j)

print(c1==4)
print(c2==4.)
print(abs(c3-0.18345697474330172-0.9830277404112437j)<1e-15)
print(c4==0-8j)
print(c5==16+0j)
print(c6==0.0625)
print(c7==-16.0)
print(abs(c8+1.9036539387158786)<1e-15)

print(d1==-12)
print(d2==-14)
print(d3==12)
print(d4==16777216)

print(e1==-2.)
print(e2==38.)
print(e3==0.)
print(e4==-98)
print(abs(e5-88980.47159826607)<1e-16)

print(f1==0.5)
print(f2==0.5)
print(f3==2)
print(f4==2)
print(f5==2.)
print(f6==23.)
print(f7==23)
print(f8==-12)
print(f9==96)
print(f10==2)


