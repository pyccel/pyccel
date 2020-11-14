@template('z', types=['int', 'real'])
@template('y', types=['int', 'real'])
@types('z', 'z', 'y')
def multi_tmplt_1(x, y, z): 
    return x + y + z

def tst_multi_tmplt_1():
    x = multi_tmplt_1(5, 5.3, 7)
    return x
