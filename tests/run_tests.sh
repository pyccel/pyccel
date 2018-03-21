# this file can be used by developers to run all tests localy before travis

pytest tests/ast/test_ast.py
pytest tests/parser/test_headers.py
pytest tests/parser/test_openmp.py
pytest tests/parser/test_openacc.py
pytest tests/syntax/test_syntax.py
pytest tests/semantic/test_semantic.py
pytest tests/codegen/test_codegen.py  

# not working for python3
#pytest tests/epyccel/test_epyccel.py  

#python tests/test_pyccel.py --execute
#python tests/test_pyccel_openmp.py --openmp --execute
#python tests/test_pyccel_lapack.py --libs='blas lapack' --execute
#python tests/test_pyccel_mpi.py --compiler=mpif90
#
##python tests/test_pyccel_openacc.py --compiler=pgfortran --openacc --execute 
