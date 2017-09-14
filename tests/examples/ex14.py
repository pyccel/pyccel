n = int()
n=100
Pi_estime = double()
Pi_estime=acos(-1.0)
def f(x):
    x = double()
    z=1+x*x
    y=1/z
    return y
h = double()
h = 1.0 /n
for k in range(1,100):
    Pi_calcule = double()
    Pi_calcule = 0.0
    for i in range(1, n):
        x = h * i
        Pi_calcule = Pi_calcule + f(x)
    Pi_calcule = h * Pi_calcule
ecart = Pi_estime - Pi_calcule
print(ecart)
