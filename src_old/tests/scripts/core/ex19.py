a   = zeros(65,int)
n   = len(a)
p   = a[n-1]
dp  = 0.0
ddp = 0.0
x   = 0.1
for i in range(1,n):
    ddp = ddp*x + 2.0*dp
    dp = dp*x + p
    p = p*x + a[n-i-1]
print(p,dp,ddp)
