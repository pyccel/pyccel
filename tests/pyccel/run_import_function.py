import sys

for mod in sys.argv[1:]:
    exec("from "+mod+" import test_func")
    print(test_func())
