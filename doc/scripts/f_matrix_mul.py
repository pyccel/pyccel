from pyccel.codegen.utilities import execute_pyccel
code = execute_pyccel('scripts/matrix_mul.py', convert_only=True)
print(code)
