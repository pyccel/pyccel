# coding: utf-8
import os

language = "--language=fortran"
compiler = "--compiler=gfortran"
execute  = ""
show     = ""
openmp   = "--openmp"

not_working = ["ex7.py"]

examples = ["matrix_product.py"]
for i in range(1, 7 + 1):
    example = "ex{0}.py".format(str(i))
    if not(example in not_working):
        examples.append(example)

for example in examples:
    print("===== running example {0} =====".format(example))
    filename = "tests/examples/openmp/{0}".format(example)
    cmd = "pyccel {0} {1} {2} {3} {4} {5}".format(filename, language, \
                                                  compiler, openmp, \
                                                  execute, show)
    os.system(cmd)
