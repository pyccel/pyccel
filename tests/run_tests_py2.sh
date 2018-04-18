# this file can be used by developers to run all tests localy before travis

python tests/ast/test_ast.py
python tests/parser/test_headers.py
python tests/parser/test_openmp.py
python tests/parser/test_openacc.py
python tests/syntax/test_syntax.py
python tests/semantic/test_semantic.py
python tests/codegen/test_codegen.py  
python tests/blas/test_blas.py  
python tests/lapack/test_lapack.py  
python tests/mpi/test_mpi.py  
python tests/openmp/test_openmp.py  

cd tests/epyccel/ ; python test_epyccel.py ; cd ../.. 
