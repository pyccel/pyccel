## module run_kut4
#''' X,Y = integrate(F,x,y,xStop,h).
#    4th-order Runge-Kutta method for solving the
#    initial value problem {y}' = {F(x,{y})}, where
#    {y} = {y[0],y[1],...y[n-1]}.
#    x,y   = initial conditions
#    xStop = terminal value of x
#    h     = increment of x used in integration
#    F     = user-supplied function that returns the
#            array F(x,y) = {y'[0],y'[1],...,y'[n-1]}.
#'''
def F(x,y):
    y=x+y
    return y
def run_kut4(x,y,h):
    x=float()
    y=float()
    h=float()  
    K0 = h*F(x,y)
    K1 = h*F(x + h/2.0, y + K0/2.0)
    K2 = h*F(x + h/2.0, y + K1/2.0)
    K3 = h*F(x + h, y + K2)
    z=K0/6.0 + 2.0*K1/6.0 + 2.0*K2/6.0 + K3/6.0
    return z
def integrate(x,y,xStop,h):
    x=float()
    y=float()
    xStop=float()
    h=float()
    n=100
    i=0
    X = zeros(shape=n,dtype=float)
    Y = zeros(shape=n,dtype=float)
    X[0]=x
    Y[0]=y
    
    while x < xStop and i<n:
        h = min(h,xStop - x)
        y = y + run_kut4(x,y,h)
        x = x + h
        X[i]=x
        Y[i]=y
        i=i+1
    return X,Y


