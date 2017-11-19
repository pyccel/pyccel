# coding: utf-8
import os

language = "--language=fortran"
compiler = "--compiler=gfortran"
execute  = "--execute"
show     = ""

not_working = []
for i in ['11', '15', '18', '20', '21']:
    ex = "ex{}.py".format(str(i))
    not_working.append(ex)

examples = []
for i in range(1, 21 + 1):
    example = "ex{0}.py".format(str(i))
    if not(example in not_working):
        examples.append(example)

for example in examples:
    print("===== running example {0} =====".format(example))
    filename = "tests/examples/{0}".format(example)
    cmd = "pyccel {0} {1} {2} {3} {4}".format(filename, language, compiler, execute, show)
    os.system(cmd)
