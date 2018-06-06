a = [6]*10
s1 = sum(a[i] for i in range(len(a)))
s2 = sum(i for i in a)
b = max(i if i>k else k for i in range(5) for k in range(10))
c = min(k if i>k else 0 if i==k else i for i in range(5) for k in range(10))

print s1,s2,b,c
 
