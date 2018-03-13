def helloworld():
    print('hello world')

#$ header function incr(int)
def incr(x):
    x = x + 1

#$ header function decr(int) results(int)
def decr(x):
    y = x - 1
    return y

#$ header function incr_array(int [:])
def incr_array(x):
    x = x + 1

#$ header function decr_array(int [:]) results(int [:])
def decr_array(x):
    y = x - 1
    return y

#$ header function f1(int, int, int) results(int)
def f1(x, n=2, m=3):
    y = x - n*m
    return y

print('Improve this test')
y = decr(2)
y = f1(1)

