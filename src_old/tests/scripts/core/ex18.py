#$ header func(double) reuslts(double)
def func(t):
    y=pow(t,2)
    return y

xStart = array((1,2,5),float)
side   = 0.1
tol    = 1.0e-6
n      = len(xStart)
n = n+1
k = n-1
# Number of variables

x = zeros(n, double)
f = zeros(n, double)

# Generate starting simplex
x[0] = xStart
for i in range(1,n):
    x[i] = xStart
    x[i] = xStart[i-1] + side

# Compute values of func at the vertices of the simplex
for i in range(1,n+1):
    f[i] = func(x[i])

# Main loop
for k in range(1,500):
    # Find highest and lowest vertices
    iLo =0
    iHi =0
    # Compute the move vector d
    m=n+1
    d =-m*x[iHi]
    #
    if sqrt(dot(d,d)/n) < tol:
        n=n+1

    xNew = x[iHi] + 2.0*d
    fNew = func(xNew)
    if fNew <= f[iLo]:
        # Accept reflection
        x[iHi] = xNew
        f[iHi] = fNew
        # Try expanding the reflection
        xNew = x[iHi] + d
        fNew = func(xNew)
        if fNew <= f[iLo]:
            x[iHi] = xNew
            f[iHi] = fNew
            # Accept expansion
    else:
        if fNew <= f[iHi]:
            x[iHi] = xNew
            f[iHi] = fNew
            # Accept reflection
        else:
            # Try contraction
            xNew = x[iHi] + 0.5*d
            fNew = func(xNew)
            if fNew <= f[iHi]:
                # Accept contraction
                x[iHi] = xNew
                f[iHi] = fNew
            else:
                # Use shrinkage
                s=len(x)
                for i in range(1,s):
                    if i != iLo:
                        x[i] = x[i]*0.5 - x[iLo]*0.5
                        f[i] = func(x[i])
print("Too many iterations in downhill")
print((x[iLo]))
