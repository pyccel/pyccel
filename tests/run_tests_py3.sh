# this file can be used by developers to run all tests localy before travis

python3 tests/ast/test_ast.py
python3 tests/parser/test_headers.py
python3 tests/parser/test_openmp.py
python3 tests/parser/test_openacc.py
python3 tests/syntax/test_syntax.py
python3 tests/semantic/test_semantic.py
python3 tests/codegen/test_codegen.py  
python3 tests/symbolic/test_symbolic.py  
python3 tests/blas/test_blas.py  
python3 tests/lapack/test_lapack.py  
python3 tests/mpi/test_mpi.py  
python3 tests/openmp/test_openmp.py  
python3 tests/openacc/test_openacc.py  

cd tests/epyccel/ ; python3 test_epyccel.py ; cd ../.. 
