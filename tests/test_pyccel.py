# coding: utf-8
from pyccel.commands.console import pyccel

ignored = [11, 15, 18, 20, 21]

for i in range(1, 21):
    filename = 'tests/examples/ex{0}.py'.format(str(i))
    if not(i in ignored):
        pyccel(files=[filename])
        print(' testing {0}: done'.format(str(i)))
