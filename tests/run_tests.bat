python -m pytest ../tests/preprocess/test_preprocess.py
python -m pytest ../tests/parser/test_headers.py
python -m pytest ../tests/parser/test_openmp.py
python -m pytest ../tests/parser/test_openacc.py
python -m pytest ../tests/syntax/test_syntax.py
python -m pytest ../tests/errors/test_errors.py
python -m pytest ../tests/warnings/test_warnings.py
python -m pytest ../tests/semantic/test_semantic.py
python -m pytest ../tests/codegen/fcode/test_fcode_codegen.py
python -m pytest ../tests/codegen/pycode/test_pycode_codegen.py
python -m pytest ../tests/complexity/test_complexity.py
python -m pytest ../tests/epyccel -v -m "not parallel"
python -m pytest ../tests/pyccel -v 
mpiexec -n 4 python -m pytest ../tests/epyccel/test_epyccel_mpi_modules.py -v -m parallel
python ../tests/internal/test_internal.py
python ../tests/macro/test_macro.py
