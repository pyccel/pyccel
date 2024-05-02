from pyccel.codegen.utilities import execute_pyccel
code = execute_pyccel('scripts/helloworld.py', convert_only=True)
print(code)
