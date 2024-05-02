## module rootsearch
#''' x1,x2 = rootsearch(f,a,b,dx).
#    Searches the interval (a,b) in increments dx for
#    the bounds (x1,x2) of the smallest root of f(x). 
#    Returns x1 = x2 = None if no roots were detected.
#'''
#
#def rootsearch(f,a,b,dx):
#$header function f(double) results(double)
def f(x):
    y=x
    return y
a=-0.5
b=2.0
dx=0.001
x1 = a
f1 = f(a)
x2 = a + dx
f2 = f(x2)
while sign(f1) == sign(f2):
    if x1  >=  b: 
        break
    x1 = x2
    f1 = f2
    x2 = x1 + dx
    f2 = f(x2)
print(( x1,x2))
