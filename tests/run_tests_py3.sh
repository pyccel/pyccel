# this file can be used by developers to run all tests localy before travis

python3 tests/parser/test_headers.py
python3 tests/parser/test_openmp.py
python3 tests/parser/test_openacc.py
python3 tests/syntax/test_syntax.py
python3 tests/semantic/test_semantic.py
python3 tests/codegen/test_codegen.py  
python3 tests/errors/test_errors.py  
python3 tests/warnings/test_warnings.py  
python3 tests/preprocess/test_preprocess.py
python3 tests/internal/test_internal.py  

cd tests/epyccel/ ; python3 test_epyccel.py ; cd ../.. 
