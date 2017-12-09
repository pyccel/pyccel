#$ header f(double) results(double)
def f(x):
    y=pow(x,3)
    return y

tol = 0.000001
x1  = -1.0
x2  = 1.0
f1  = f(x1)
f2  = f(x2)
f3  = f1
n   = ceil(log(abs(x2 - x1)/tol)/log(2.0))
while f3 > tol:
    x3 = 0.5*x1 + 0.5*x2
    f3 = f(x3)
    if f1*f3<0:
        x1 = x3
        f1 = f3
    if f2*f3<0:
        x2 = x3
        f2 = f3

