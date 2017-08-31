# coding: utf-8
import os

language = "--language=fortran"
compiler = "--compiler=gfortran"
execute  = ""
show     = "--show"

not_working = ["ex7.py","ex11.py"]

examples = []
for i in range(1, 12 + 1):
    example = "ex{0}.py".format(str(i))
    if not(example in not_working):
        examples.append(example)

for example in examples:
    print "===== running example {0} =====".format(example)
    filename = "--filename=tests/examples/{0}".format(example)
    cmd = "python tests/pyccel_cmd.py {0} {1} {2} {3} {4}".format(filename, language,
                                                            compiler, execute, show)
    os.system(cmd)