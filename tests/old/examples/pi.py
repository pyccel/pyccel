from numpy import arccos
n=100
 

Pi_estime = 3.14
#Pi_estime=arccos(-1.0)

#def f(x):
#    return 1/(1+x*x)

  
h = 1.0 /n



  
for k in range(1,100):     
    Pi_calcule = 0.0
    for i in range(1, n):
        x = h * i
        Pi_calcule = Pi_calcule + 4.0 /( 1.0+x*x)
     
    Pi_calcule = h * Pi_calcule

ecart = Pi_estime - Pi_calcule
print(ecart)
