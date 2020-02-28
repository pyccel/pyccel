# This file can be used by developers to run all tests localy before travis
# It must be run from pyccel's root directory, which contains 'pyccel' and 'tests'

#python3 tests/parser/test_headers.py
#python3 tests/parser/test_openmp.py
#python3 tests/parser/test_openacc.py
#python3 tests/syntax/test_syntax.py
#python3 tests/semantic/test_semantic.py
#python3 tests/codegen/test_codegen.py  
#python3 tests/errors/test_errors.py  
#python3 tests/warnings/test_warnings.py  
#python3 tests/preprocess/test_preprocess.py
#python3 tests/internal/test_internal.py 
#python3 tests/external/test_external.py  
#python3 tests/macro/test_macro.py  

python3 -m pytest tests/epyccel/test_epyccel_functions.py
python3 -m pytest tests/epyccel/test_epyccel_modules.py
python3 -m pytest tests/epyccel/test_arrays.py
python3 -m pytest tests/epyccel/test_loops.py
python3 -m pytest tests/epyccel/test_kind.py
python3 -m pytest tests/pyccel/test_pyccel.py
mpirun -n 4 python3 -m pytest tests/epyccel/test_epyccel_mpi_modules.py
