from pyccel.codegen.utilities import execute_pyccel
code = execute_pyccel('../scripts/Point.py', convert_only=True)
print(code)
