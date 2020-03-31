#!/bin/bash

INSTALL_DIR=$(python -c "import pyccel; print(pyccel.__path__[0])")
SITE_DIR=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
printf "import coverage; coverage.process_startup()" > ${SITE_DIR}/pyccel_cov.pth
printf "[run]\nparallel = True\nsource = ${INSTALL_DIR}\n[report]\ninclude = ${INSTALL_DIR}/*\n[xml]\noutput = cobertura.xml" > .coveragerc
export COVERAGE_PROCESS_START=$(pwd)/.coveragerc

python -m pytest tests/preprocess/test_preprocess.py
python -m pytest tests/parser/test_headers.py
python -m pytest tests/parser/test_openmp.py
python -m pytest tests/parser/test_openacc.py
python -m pytest tests/syntax/test_syntax.py
python -m pytest tests/errors/test_errors.py
python -m pytest tests/warnings/test_warnings.py
python -m pytest tests/semantic/test_semantic.py
python -m pytest tests/codegen/fcode/test_fcode_codegen.py
python -m pytest tests/codegen/pycode/test_pycode_codegen.py
python -m pytest tests/complexity/test_complexity.py
python -m pytest tests/epyccel -v -m "not parallel"
python -m pytest tests/pyccel -v

mpirun -n 4 python -m pytest tests/epyccel/test_epyccel_mpi_modules.py -v -m parallel
#mpirun -n 4 python -m pytest tests/epyccel -v -m parallel
# this test must be executed with python, since we are calling mpif90 inside it

python tests/internal/test_internal.py
#python tests/external/test_external.py
python tests/macro/test_macro.py
#python tests/test_pyccel.py --execute
#python tests/test_pyccel_openmp.py --openmp --execute
#python tests/test_pyccel_lapack.py --libs='blas lapack' --execute
#python tests/test_pyccel_mpi.py --compiler=mpif90

coverage combine
coverage xml
rm ${SITE_DIR}/pyccel_cov.pth
