from pyccel.epyccel import epyccel

def openmp_ex1():
    return 1

f1 = epyccel(openmp_ex1, accelerator='openmp', language='c')

print(f1())
