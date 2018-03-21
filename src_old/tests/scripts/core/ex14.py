n=100
#Pi_estime=acos(-1.0) # TODO not available yet

#$ header f(double) results(double)
def f(x):
    y = 1.0/(1.0 + x*x)
    return y

#TODO fix h is processed as int
h = 1.0/n
for k in range(1,1000):
    Pi_calcule = 0.0
    for i in range(1, n):
        x = h * i
        Pi_calcule = Pi_calcule + f(x)
    Pi_calcule = h * Pi_calcule
print(Pi_calcule)
#ecart = Pi_estime - Pi_calcule
#print(ecart)

