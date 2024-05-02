# pylint: disable=missing-function-docstring, missing-module-docstring
def mod_values(x : int):
    return x+2,x+3

if __name__ == '__main__':
    a = 3
    b = 4
    c = 5
    a,d = mod_values(a)
    b,e = mod_values(b)
    c,f = mod_values(c)

    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)
