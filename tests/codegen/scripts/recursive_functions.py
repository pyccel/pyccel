#$ header fact(int) results(int)
def fact(n):
   if n == 0:
       z = 1
       return z
   else:
       z = n*fact(n-1)
       return z

print(fact(5))


