n=100
Pi_estime=acos(-1.0)

#$ header f(double)
def f(x):
    z = 1.0 + x*x
    y = 1.0/z
    return y

h = 1.0/n
for k in range(1,100):
    Pi_calcule = 0.0
    for i in range(1, n):
        x = h * i
        Pi_calcule = Pi_calcule + f(x)
    Pi_calcule = h * Pi_calcule
ecart = Pi_estime - Pi_calcule
print(ecart)
