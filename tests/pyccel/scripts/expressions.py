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

d1 = 1-1-2-4*3+7-5
d2 = 1-1-2**2.-4*3+7-5
d3 = 3*(1+4//2*3.-3.)
d4 = (2*2)**(3*4)

e1 = 23*4%16.032/16.4//2./2%3 + 34 + 23*23*4%16.0+32/16.4//2./2%3
e2 = (1%1**4/23**2/6//2.)
e3 = 2 * 4 + 3**4 - 3*5**3//2
e4 = (1%16.032/16.032//25.9948+25.9948*25.99-(25.9-33.74//33.746134%33.746139653844324-33.7464%36.13899562242578)-36.13899562242578*\
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

print(d1==-12)
print(d2==-14)
print(d3==12)
print(d4==16777216)

print(e1==38.)
print(e2==0.)
print(e3==-98)
print(abs(e4-88980.47159826607)<1e-16)



