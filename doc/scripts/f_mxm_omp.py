from pyccel.codegen.utilities import execute_pyccel
code = execute_pyccel('scripts/mxm_omp.py', convert_only=True)
print(code)
