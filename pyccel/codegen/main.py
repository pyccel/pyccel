#!/usr/bin/python3

from pyccel.codegen.pipeline import execute_pyccel

# pyccel/codegen/
def main() :
    file = "pyccel/codegen/titi.py"
    execute_pyccel(file, language="c", verbose="true")
if __name__ == "__main__" :
    main() 
