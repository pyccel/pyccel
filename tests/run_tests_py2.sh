# this file can be used by developers to run all tests localy before travis

python tests/parser/test_headers.py
python tests/parser/test_openmp.py
python tests/parser/test_openacc.py
python tests/syntax/test_syntax.py
python tests/semantic/test_semantic.py
python tests/codegen/test_codegen.py  
python tests/errors/test_errors.py  
python tests/warnings/test_warnings.py  
python tests/preprocess/test_preprocess.py
python tests/internal/test_internal.py
python tests/external/test_external.py  
python tests/macro/test_macro.py

cd tests/epyccel/ ; python test_epyccel.py ; cd ../.. 
