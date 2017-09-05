def A(x):
    y=x[1]+x[2]+x[3]
    z=x[1]+x[2]+x[3]
    t=x[1]+x[2]+x[3]
    return y,z,t
    
x=array([0,0,0],dtype=float)
b=array([2,5,-1],dtype=float)
tol=1.0e-9
n = len(b)
r = b - A(x)
s = r
alpha=0.5
ss=dot(r,r)
for i in range(1,n):
    u = 2.0*x
    l=dot(s,u)
    ll=dot(s,r)
    alpha = ll/l
    x = x + alpha*s
    r = b - 2.0*x
    ss=dot(r,r)
    l=dot(s,u)
    ll=dot(r,u)
    beta = -ll/l
    s = r + beta*s
print(x,i)
