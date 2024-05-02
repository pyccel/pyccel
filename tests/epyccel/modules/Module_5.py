# pylint: disable=missing-function-docstring, missing-module-docstring

def nothing(x  : 'int' =  None):
    if x is None :
        return 2
    return x

def add(x  : 'int' =  None, y  : 'int' =  None):
    a = 0
    b = 0
    if x is not None :
        a = x
    if y is not None :
        b = y
    return a + b


def call_optional_1(x : 'int'):
    a = nothing(x + 2)
    return a

def call_optional_2():
    a = nothing(2)
    return a

def call_optional_3(x : 'int'):
    a = 2 + nothing(x)
    return a

def call_optional_4(x : 'int'):
    a = add(x, 2)
    return a

def call_optional_5(x : 'int'):
    a = add(x + 2, 2)
    return a

def call_optional_6():
    a = add(2, 2)
    return a

def call_optional_7():
    a = add(2)
    return a

def call_optional_8():
    a = add()
    return a


def call_optional_9():
    if nothing(2) :
        return 2
    return 1

def call_optional_10():
    if add(2, 2) :
        return 2
    return 1

def call_optional_11():
    a = 0
    i = 0
    while i < nothing(3) :
        i = i + 1
        a = a + nothing(3)
    return a

def call_optional_12():
    a = 0
    for i in range(0, nothing(3)) : # pylint: disable=unused-variable
        a = a + nothing(3)
    return a

