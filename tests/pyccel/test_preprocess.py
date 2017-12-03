# coding: utf-8

code = '''
n = 10
for i in range(0,n):
    x = 2 * i

    y = x / 3
    # a comment
    if y > 1:

        print(y)

        for j in range(0, 3):
            x = x * y

            y = x + 1

if x > 1:
    print(x)
'''

code = '''
#$ header legendre(int)
def legendre(p):
    k = p + 1
    x = zeros(k, double)
    w = zeros(k, double)
    if p == 1:
        x[0] = -0.577350269189625765
        x[1] =  0.577350269189625765
        w[0] =  1.0
        w[1] =  1.0
    elif p == 2:
        x[0] = -0.774596669241483377
        x[1] = 0.0
        x[2] = 0.774596669241483377
        w[0] = 0.55555555555555556
        w[1] = 0.888888888888888889
        w[2] = 0.55555555555555556
    elif p == 3:
        x[0] = -0.861136311594052575
        x[1] = -0.339981043584856265
        x[2] = 0.339981043584856265
        x[3] = 0.861136311594052575
        w[0] = 0.347854845137453853
        w[1] = 0.65214515486254615
        w[2] = 0.65214515486254614
        w[3] = 0.34785484513745386
    return x,w

#$ comment
if x > 1:
    print(x)
'''

from pyccel.codegen import preprocess_as_str
txt = preprocess_as_str(code)
print(txt)
