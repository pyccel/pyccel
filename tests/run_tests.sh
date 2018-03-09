# this file can be used by developers to run all tests localy before travis

pytest tests/pyccel/ast/test_ast.py
pytest tests/pyccel/parser/test_imports.py
pytest tests/pyccel/parser/test_headers.py
pytest tests/pyccel/parser/test_openmp.py
pytest tests/pyccel/parser/test_openacc.py
pytest tests/pyccel/parser/test_syntax.py
pytest tests/pyccel/parser/test_semantic.py
pytest tests/pyccel/codegen/test_codegen.py  

#python tests/test_pyccel.py --execute
#python tests/test_pyccel_openmp.py --openmp --execute
#python tests/test_pyccel_lapack.py --libs='blas lapack' --execute
#python tests/test_pyccel_mpi.py --compiler=mpif90
#
##python tests/test_pyccel_openacc.py --compiler=pgfortran --openacc --execute 
