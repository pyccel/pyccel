#def evalPoly(a,x):
## module evalPoly
# p,dp,ddp = evalPoly(a,x).
#    Evaluates the polynomial
#    p = a[0] + a[1]*x + a[2]*x^2 +...+ a[n]*x^n
#    with its derivatives dp = p' and ddp = p"
#    at x.
a=zeros(65,int)
x=float()
n = len(a)
p = a[n-1]
dp = 0.0
ddp =0.0
for i in range(1,n):
    ddp = ddp*x + 2.0*dp
    dp = dp*x + p
    p = p*x + a[n-i-1]
#return
print(p,dp,ddp)
