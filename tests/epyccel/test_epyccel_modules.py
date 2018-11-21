from pyccel.epyccel import epyccel

def test_module_1():
    import modules.Module_1 as mod

    modnew = epyccel(mod)

    from numpy import zeros
    x = zeros(5)
    modnew.f(x)
    modnew.g(x)
    print(x)

##################################"
if __name__ == '__main__':
    test_module_1()
