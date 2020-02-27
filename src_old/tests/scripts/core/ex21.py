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
#TODO improve inout hadeling
#TODO fix comment bug when it's not properly alligned in a blog
#TODO fix the bug of two variables with the same name but one is lower case and the other is upper



#$ header function F(double,double) results(double)
#$ header function run_kut4(double,double,double) results(double)
#$ header function integrate(double,double,double,double) results(*double[:],*double[:])

def F(x,y):
    z=x+y
    return z
def run_kut4(x,y,h):
    K0 = h*F(x,y)
    K1 = h*F(x + h/2.0, y + K0/2.0)
    K2 = h*F(x + h/2.0, y + K1/2.0)
    K3 = h*F(x + h, y + K2)
    z=K0/6.0 + 2.0*K1/6.0 + 2.0*K2/6.0 + K3/6.0
    return z
def integrate(a,b,xStop,h):
    n=100
    i=0
    X = zeros(n,double)
    Y = zeros(n,double)
    X[0]=a
    Y[0]=b

    while X[i] < xStop and i<n:
        Y[i] = Y[i] + run_kut4(X[i],Y[i],h)
        X[i] = X[i]+h
        i=i+1
    return X,Y
