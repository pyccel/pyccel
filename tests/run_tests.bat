REM python -m pytest ../tests/preprocess/test_preprocess.py
REM python -m pytest ../tests/parser/test_headers.py
REM python -m pytest ../tests/parser/test_openmp.py
REM python -m pytest ../tests/parser/test_openacc.py
REM python -m pytest ../tests/syntax/test_syntax.py
REM python -m pytest ../tests/errors/test_errors.py
REM python -m pytest ../tests/warnings/test_warnings.py
REM python -m pytest ../tests/semantic/test_semantic.py
REM python -m pytest ../tests/codegen/fcode/test_fcode_codegen.py
REM python -m pytest ../tests/codegen/pycode/test_pycode_codegen.py
REM python -m pytest ../tests/complexity/test_complexity.py
REM python -m pytest ../tests/epyccel -v -x -m "not parallel"
python -m pytest ../tests/pyccel -v -x
REM TODO mpirun -n 4 python -m pytest ../tests/epyccel/test_epyccel_mpi_modules.py -v -x -m parallel
python ../tests/internal/test_internal.py -v -x
python ../tests/macro/test_macro.py -v -x
